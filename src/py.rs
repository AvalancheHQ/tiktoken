use std::collections::HashSet;

use pyo3::{
    IntoPyObjectExt, PyResult,
    buffer::{Element, PyBuffer, PyUntypedBuffer},
    exceptions,
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
        // Fast path for the common case where `tokens` is a concrete `list`:
        // resolve every token to its byte slice, then copy those bytes exactly
        // once, straight into the destination Python `bytes` object. Avoiding
        // the intermediate `Vec<Rank>` and `Vec<u8>` (i.e. that single copy) is
        // the win, not the traversal count — this still walks the tokens twice.
        //
        // Note: unlike the non-`list` path, this holds the GIL for the whole
        // decode because it reads Python list items throughout. That is the
        // right trade single-threaded, but it does not release the GIL the way
        // `py.detach` does, so multi-threaded decoding on a GIL build won't
        // overlap here.
        if let Ok(list) = tokens.downcast::<PyList>() {
            // Resolve every token to its byte slice and accumulate total length.
            let mut slices: Vec<&[u8]> = Vec::with_capacity(list.len());
            let mut total_len = 0usize;
            for item in list.iter() {
                let token = read_rank(&item)?;
                let token_bytes = self.token_bytes(token).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "Invalid token for decoding: {token}"
                    ))
                })?;
                total_len += token_bytes.len();
                slices.push(token_bytes);
            }
            // Copy the resolved bytes directly into the final buffer.
            let bytes = PyBytes::new_with(py, total_len, |mut buf| {
                for s in &slices {
                    buf[..s.len()].copy_from_slice(s);
                    buf = &mut buf[s.len()..];
                }
                Ok(())
            })?;
            return Ok(bytes.into());
        }

        // Fast path for token ids that live in a contiguous integer buffer:
        // `numpy` arrays, torch CPU tensors, `array.array`, `memoryview`. The
        // generic path below reaches those through the sequence protocol, which
        // materialises (and immediately throws away) one Python `int` per token;
        // here the ids are read straight out of the buffer instead.
        if unsafe { pyo3::ffi::PyObject_CheckBuffer(tokens.as_ptr()) } == 1
            && let Some(result) = self.decode_int_buffer(py, tokens)
        {
            return result;
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

impl CoreBPE {
    /// Decodes token ids read directly out of a one-dimensional, C-contiguous
    /// buffer of fixed-width integers (`numpy` arrays, torch CPU tensors,
    /// `array.array`, `memoryview`).
    ///
    /// Returns `None` when `tokens` is not such a buffer, or when one of its
    /// values does not fit a token id, so the caller falls back to the generic
    /// extraction path and its exact error behaviour is preserved.
    ///
    /// Kept out of line: inlining it (and the four `decode_from_buffer`
    /// instantiations it fans out to) into `py_decode_bytes` more than doubles
    /// that function's code size and measurably slows the `list` fast path.
    #[inline(never)]
    fn decode_int_buffer(
        &self,
        py: Python<'_>,
        tokens: &Bound<'_, PyAny>,
    ) -> Option<Result<Py<PyBytes>, PyErr>> {
        let buffer = PyUntypedBuffer::get(tokens).ok()?;
        if buffer.dimensions() > 1 {
            return None;
        }
        // The widths token ids actually arrive in: `uint32` (what
        // `encode_to_numpy` produces) and the 32/64-bit signed and unsigned
        // integers `numpy`/torch default to. Anything else falls back.
        if let Ok(buffer) = buffer.as_typed::<u32>() {
            self.decode_from_buffer(py, buffer)
        } else if let Ok(buffer) = buffer.as_typed::<i64>() {
            self.decode_from_buffer(py, buffer)
        } else if let Ok(buffer) = buffer.as_typed::<i32>() {
            self.decode_from_buffer(py, buffer)
        } else if let Ok(buffer) = buffer.as_typed::<u64>() {
            self.decode_from_buffer(py, buffer)
        } else {
            None
        }
    }

    /// Resolves the token ids in `buffer` and copies their bytes into a Python
    /// `bytes` object, mirroring the `list` fast path in `py_decode_bytes`.
    fn decode_from_buffer<T>(
        &self,
        py: Python<'_>,
        buffer: &PyBuffer<T>,
    ) -> Option<Result<Py<PyBytes>, PyErr>>
    where
        T: Element + Copy,
        Rank: TryFrom<T>,
    {
        // `None` unless the buffer is C-contiguous, in which case the ids are a
        // plain slice of integers: no per-token Python object is involved.
        let items = buffer.as_slice(py)?;

        let mut slices: Vec<&[u8]> = Vec::with_capacity(items.len());
        let mut total_len = 0usize;
        for item in items {
            // Out-of-range values (negative, or wider than a token id) fall back
            // to the generic path, which raises the original `OverflowError`.
            let token = Rank::try_from(item.get()).ok()?;
            let token_bytes = match self.token_bytes(token) {
                Some(token_bytes) => token_bytes,
                None => {
                    return Some(Err(pyo3::exceptions::PyKeyError::new_err(format!(
                        "Invalid token for decoding: {token}"
                    ))));
                }
            };
            total_len += token_bytes.len();
            slices.push(token_bytes);
        }

        Some(
            PyBytes::new_with(py, total_len, |mut buf| {
                for s in &slices {
                    buf[..s.len()].copy_from_slice(s);
                    buf = &mut buf[s.len()..];
                }
                Ok(())
            })
            .map(Into::into),
        )
    }
}

/// Read a token id from a Python object, reading a plain in-range `int` with a
/// direct CPython C-API call and otherwise falling back to pyo3's `extract`.
#[inline]
fn read_rank(item: &Bound<'_, PyAny>) -> PyResult<Rank> {
    // SAFETY: `item` is a valid, non-null borrowed Python object for the
    // duration of the call, and we hold the GIL (we have a `Bound`).
    let value = unsafe { pyo3::ffi::PyLong_AsUnsignedLong(item.as_ptr()) };
    // `PyLong_AsUnsignedLong` returns `(c_ulong)-1` on error *and* for the
    // legitimate value `c_ulong::MAX`, so the sentinel must always route to the
    // slow path, which disambiguates the two.
    if value != std::os::raw::c_ulong::MAX
        && let Ok(rank) = Rank::try_from(value)
    {
        return Ok(rank);
    }
    // Fall back to `extract` to preserve the exact original error types
    // (`OverflowError` for negative / out-of-range, `TypeError` for non-int).
    // `PyErr_Clear` is a no-op when the sentinel was the legitimate value.
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
