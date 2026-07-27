use std::collections::HashSet;

use pyo3::{
    IntoPyObjectExt, PyResult, exceptions,
    prelude::*,
    pybacked::PyBackedStr,
    types::{PyBytes, PyList},
};
use rustc_hash::FxHashMap as HashMap;

use crate::{CoreBPE, Rank, byte_pair_encode};

#[pymethods]
impl CoreBPE {
    #[new]
    fn py_new(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> PyResult<Self> {
        Self::new_internal(encoder, special_tokens_encoder, pattern)
            .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.to_string()))
    }

    // ====================
    // Encoding
    // ====================

    #[pyo3(name = "encode_ordinary")]
    fn py_encode_ordinary(&self, py: Python, text: &str) -> Vec<Rank> {
        py.detach(|| self.encode_ordinary(text))
    }

    #[pyo3(name = "encode")]
    fn py_encode(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<Vec<Rank>> {
        py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            match self.encode(text, &allowed_special) {
                Ok((tokens, _)) => Ok(tokens),
                Err(e) => Err(PyErr::new::<exceptions::PyValueError, _>(e.message)),
            }
        })
    }

    fn encode_to_tiktoken_buffer(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<Py<PyAny>> {
        let tokens_res = py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            self.encode(text, &allowed_special)
        });

        let tokens = match tokens_res {
            Ok((tokens, _)) => tokens,
            Err(e) => return Err(PyErr::new::<exceptions::PyValueError, _>(e.message)),
        };

        let buffer = TiktokenBuffer { tokens };
        buffer.into_py_any(py)
    }

    fn _encode_bytes(&self, py: Python, bytes: &[u8]) -> Vec<Rank> {
        py.detach(|| {
            match std::str::from_utf8(bytes) {
                // Straightforward case
                Ok(text) => self.encode_ordinary(text),
                // Oops, don't actually have UTF-8. But we need to do the regex splitting in
                // Unicode space, so we make our best guess at where we would have splits
                Err(e) => {
                    let text = unsafe { std::str::from_utf8_unchecked(&bytes[..e.valid_up_to()]) };
                    let (tokens, last_piece_token_len) =
                        self.encode(text, &HashSet::new()).unwrap();
                    let (mut tokens, last_piece_token_len) =
                        self._increase_last_piece_token_len(tokens, last_piece_token_len);

                    let mut unstable_bytes;
                    if !tokens.is_empty() && last_piece_token_len > 0 {
                        // Lop off the tokens from the last piece and run BPE on the remaining bytes
                        // This likely matches what models see better, e.g. if you assume we're
                        // dealing with truncated UTF-8 bytes.
                        // Niche, but note this may not be correct if we'd have had a regex
                        // split between the valid UTF-8 and the invalid bytes.
                        unstable_bytes = self
                            .decode_bytes(&tokens[tokens.len() - last_piece_token_len..])
                            .unwrap();
                        unstable_bytes.extend_from_slice(&bytes[e.valid_up_to()..]);

                        tokens.truncate(tokens.len() - last_piece_token_len);
                    } else {
                        unstable_bytes = bytes[e.valid_up_to()..].to_vec();
                    }

                    if !unstable_bytes.is_empty() {
                        match self.encoder.get(&unstable_bytes) {
                            Some(token) => tokens.push(*token),
                            None => {
                                tokens.extend(&byte_pair_encode(&unstable_bytes, &self.encoder))
                            }
                        }
                    }
                    tokens
                }
            }
        })
    }

    #[pyo3(name = "encode_with_unstable")]
    fn py_encode_with_unstable(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<(Vec<Rank>, Py<PyList>)> {
        let (tokens, completions): (Vec<Rank>, HashSet<Vec<Rank>>) = py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            self._encode_unstable_native(text, &allowed_special)
        });
        let py_completions = PyList::new(py, completions.into_iter())?;
        Ok((tokens, py_completions.into()))
    }

    fn encode_single_token(&self, piece: &[u8]) -> PyResult<Rank> {
        if let Some(token) = self.encoder.get(piece).copied() {
            return Ok(token);
        }
        if let Ok(piece_str) = std::str::from_utf8(piece) {
            if let Some(token) = self.special_tokens_encoder.get(piece_str).copied() {
                return Ok(token);
            }
        }
        Err(PyErr::new::<exceptions::PyKeyError, _>(piece.to_owned()))
    }

    fn encode_single_piece(&self, piece: &[u8]) -> Vec<Rank> {
        if let Some(token) = self.encoder.get(piece) {
            return vec![*token];
        }
        byte_pair_encode(piece, &self.encoder)
    }

    // ====================
    // Decoding
    // ====================

    #[pyo3(name = "decode_bytes")]
    fn py_decode_bytes(&self, py: Python, tokens: &Bound<'_, PyAny>) -> Result<Py<PyBytes>, PyErr> {
        // Decode is dominated by two costs that the profile makes clear on the
        // large-token benchmarks:
        //   1. reading every Python `int` token out of the list, and
        //   2. copying the decoded bytes around.
        //
        // For the common case where `tokens` is a concrete `list`, do both in a
        // single fused pass with no intermediate `Vec<Rank>` and no intermediate
        // `Vec<u8>`:
        //   * read each token with the CPython C-API (`PyLong_AsUnsignedLong`),
        //     which avoids pyo3's generic `FromPyObject::<u32>::extract`
        //     machinery (per-element type dispatch + error scaffolding),
        //   * resolve each token to its decoder byte slice immediately, and
        //   * write those bytes straight into the destination Python `bytes`
        //     object via `PyBytes::new_with`, so the decoded bytes are copied
        //     exactly once, into their final buffer.
        //
        // Anything that is not a plain in-range `int` (negative, out of range,
        // the `ULONG_MAX` error sentinel, or a non-int) clears the pending
        // error and falls back to the generic `extract`, so token error
        // behaviour (`OverflowError` / `TypeError`) is exactly preserved. A
        // token that is not a known id still raises `KeyError` identically.
        if let Ok(list) = tokens.downcast::<PyList>() {
            // First pass: resolve every token to its byte slice and total length.
            let mut slices: Vec<&[u8]> = Vec::with_capacity(list.len());
            let mut total_len = 0usize;
            for item in list.iter() {
                let token = read_rank(&item)?;
                let token_bytes = match self.decoder.get(&token) {
                    Some(bytes) => bytes.as_slice(),
                    None => match self.special_tokens_decoder.get(&token) {
                        Some(bytes) => bytes.as_slice(),
                        None => {
                            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                                "Invalid token for decoding: {token}"
                            )));
                        }
                    },
                };
                total_len += token_bytes.len();
                slices.push(token_bytes);
            }
            // Second pass: write the bytes directly into the final buffer.
            let bytes = PyBytes::new_with(py, total_len, |mut buf| {
                for s in &slices {
                    buf[..s.len()].copy_from_slice(s);
                    buf = &mut buf[s.len()..];
                }
                Ok(())
            })?;
            return Ok(bytes.into());
        }

        // Non-`list` inputs keep the original generic extraction path.
        let tokens: Vec<Rank> = tokens.extract()?;
        match py.detach(|| self.decode_bytes(&tokens)) {
            Ok(bytes) => Ok(PyBytes::new(py, &bytes).into()),
            Err(e) => Err(pyo3::exceptions::PyKeyError::new_err(format!("{}", e))),
        }
    }

    fn decode_single_token_bytes(&self, py: Python, token: Rank) -> PyResult<Py<PyBytes>> {
        if let Some(bytes) = self.decoder.get(&token) {
            return Ok(PyBytes::new(py, bytes).into());
        }
        if let Some(bytes) = self.special_tokens_decoder.get(&token) {
            return Ok(PyBytes::new(py, bytes).into());
        }
        Err(PyErr::new::<exceptions::PyKeyError, _>(token.to_string()))
    }

    // ====================
    // Miscellaneous
    // ====================

    fn token_byte_values(&self, py: Python) -> Vec<Py<PyBytes>> {
        self.sorted_token_bytes
            .iter()
            .map(|x| PyBytes::new(py, x).into())
            .collect()
    }
}

/// Read a token id from a Python object.
///
/// The common case — a plain `int` that fits in a `u32` — is read with a single
/// direct CPython C-API call (`PyLong_AsUnsignedLong`), avoiding pyo3's generic
/// `FromPyObject::<u32>::extract` machinery. Anything that does not fit that
/// fast path (non-int, negative, out of range, or the ambiguous `ULONG_MAX`
/// error sentinel) clears any pending error and falls back to the generic
/// `extract`, so the resulting error type is exactly what it was before.
#[inline]
fn read_rank(item: &Bound<'_, PyAny>) -> PyResult<Rank> {
    // SAFETY: `item` is a valid, non-null borrowed Python object for the
    // duration of the call, and we hold the GIL (we have a `Bound`).
    let value = unsafe { pyo3::ffi::PyLong_AsUnsignedLong(item.as_ptr()) };
    // `PyLong_AsUnsignedLong` returns `(c_ulong)-1` and sets an exception on
    // any error (not an int, negative, or too large). It can also legitimately
    // return that value for the in-range integer `c_ulong::MAX`; in that case
    // no error is set. Treat the sentinel as "needs the slow path" either way.
    if value != std::os::raw::c_ulong::MAX
        && let Ok(rank) = Rank::try_from(value)
    {
        return Ok(rank);
    }
    // Clear any pending error from the fast path and fall back to the generic
    // extraction, which produces the original error types (`OverflowError` for
    // negative / out-of-range, `TypeError` for non-int) and handles the exact
    // `c_ulong::MAX` value correctly.
    unsafe { pyo3::ffi::PyErr_Clear() };
    item.extract::<Rank>()
}

#[pyclass(frozen)]
struct TiktokenBuffer {
    tokens: Vec<Rank>,
}

#[pymethods]
impl TiktokenBuffer {
    // Based on https://github.com/PyO3/pyo3/blob/v0.22.2/tests/test_buffer_protocol.rs#L25
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(pyo3::exceptions::PyBufferError::new_err("View is null"));
        }
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "Object is not writable",
            ));
        }
        unsafe {
            let view_ref = &mut *view;
            view_ref.obj = slf.clone().into_any().into_ptr();

            let data = &slf.borrow().tokens;
            view_ref.buf = data.as_ptr() as *mut std::os::raw::c_void;
            view_ref.len = (data.len() * std::mem::size_of::<Rank>()) as isize;
            view_ref.readonly = 1;
            view_ref.itemsize = std::mem::size_of::<Rank>() as isize;
            view_ref.format = if (flags & pyo3::ffi::PyBUF_FORMAT) == pyo3::ffi::PyBUF_FORMAT {
                let msg = std::ffi::CString::new("I").unwrap();
                msg.into_raw()
            } else {
                std::ptr::null_mut()
            };
            view_ref.ndim = 1;
            view_ref.shape = if (flags & pyo3::ffi::PyBUF_ND) == pyo3::ffi::PyBUF_ND {
                &mut view_ref.len
            } else {
                std::ptr::null_mut()
            };
            view_ref.strides = if (flags & pyo3::ffi::PyBUF_STRIDES) == pyo3::ffi::PyBUF_STRIDES {
                &mut view_ref.itemsize
            } else {
                std::ptr::null_mut()
            };
            view_ref.suboffsets = std::ptr::null_mut();
            view_ref.internal = std::ptr::null_mut();
        }

        Ok(())
    }

    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        // Note that Py_buffer doesn't have a Drop impl
        unsafe {
            let view_ref = &mut *view;
            if !view_ref.format.is_null() {
                std::mem::drop(std::ffi::CString::from_raw(view_ref.format));
            }
        }
    }
}

#[pymodule(gil_used = false)]
fn _tiktoken(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<CoreBPE>()?;
    Ok(())
}
