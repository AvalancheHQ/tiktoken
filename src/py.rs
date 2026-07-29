use std::collections::HashSet;
use std::ffi::CString;

use pyo3::{
    IntoPyObjectExt, PyResult, exceptions,
    prelude::*,
    pybacked::PyBackedStr,
    types::{PyBytes, PyList, PyString},
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
            let (slices, total_len) = self.resolve_token_slices(list)?;
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

        // Non-`list` inputs keep the original generic extraction path.
        let tokens: Vec<Rank> = tokens.extract()?;
        match py.detach(|| self.decode_bytes(&tokens)) {
            Ok(bytes) => Ok(PyBytes::new(py, &bytes).into()),
            Err(e) => Err(pyo3::exceptions::PyKeyError::new_err(format!("{}", e))),
        }
    }

    /// Decodes tokens straight into a Python `str`.
    ///
    /// `Encoding.decode` used to build an intermediate `bytes` object and then
    /// let Python re-decode it with `bytes.decode("utf-8", errors=...)`. That
    /// meant the text was materialised twice (once in the `bytes` buffer, once
    /// in the `str`), plus pyo3's defensive zero-fill of the `bytes` buffer and
    /// an extra Python-level call. Here the token bytes are written once,
    /// directly into the buffer of the `str` object that is returned.
    #[pyo3(name = "decode_str")]
    #[pyo3(signature = (tokens, errors="replace"))]
    fn py_decode_str(
        &self,
        py: Python<'_>,
        tokens: &Bound<'_, PyAny>,
        errors: &str,
    ) -> PyResult<Py<PyString>> {
        if let Ok(list) = tokens.downcast::<PyList>() {
            let (slices, total_len) = self.resolve_token_slices(list)?;
            return chunks_to_str(py, &slices, total_len, errors);
        }

        // Non-`list` inputs keep the original generic extraction path.
        let tokens: Vec<Rank> = tokens.extract()?;
        match py.detach(|| self.decode_bytes(&tokens)) {
            Ok(bytes) => chunks_to_str(py, &[bytes.as_slice()], bytes.len(), errors),
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
    /// Resolves every token of a Python `list` to its byte slice, also
    /// returning the total number of bytes they represent.
    ///
    /// The slices borrow from the tokeniser's decode tables, so no token bytes
    /// are copied here.
    #[inline]
    fn resolve_token_slices<'a>(
        &'a self,
        list: &Bound<'_, PyList>,
    ) -> PyResult<(Vec<&'a [u8]>, usize)> {
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
        Ok((slices, total_len))
    }
}

/// Concatenates `chunks` (whose lengths sum to `total_len`) into a Python `str`.
///
/// The bytes are written once, into the buffer of a freshly allocated `str`
/// object that nothing else can observe yet. Decoded text is overwhelmingly
/// ASCII, in which case the filled object *is* the answer and is returned as
/// is. Otherwise the bytes are handed to CPython's UTF-8 decoder (which applies
/// `errors`, exactly like the previous `bytes.decode("utf-8", errors=...)`) and
/// the scratch object is dropped.
fn chunks_to_str(
    py: Python<'_>,
    chunks: &[&[u8]],
    total_len: usize,
    errors: &str,
) -> PyResult<Py<PyString>> {
    if total_len == 0 {
        return Ok(PyString::new(py, "").unbind());
    }
    // SAFETY: `scratch` is a fresh, unshared `str` object of exactly
    // `total_len` bytes, so writing that many bytes into its buffer cannot
    // overflow it and cannot be observed by any other code. It is only handed
    // back to Python once its contents are known to be ASCII (and hence a
    // valid ASCII `str`); otherwise it is dropped without being exposed.
    unsafe {
        let scratch = pyo3::ffi::PyUnicode_New(total_len as pyo3::ffi::Py_ssize_t, 127);
        if scratch.is_null() {
            return Err(PyErr::fetch(py));
        }
        let buf = pyo3::ffi::PyUnicode_DATA(scratch).cast::<u8>();
        let mut offset = 0usize;
        for chunk in chunks {
            std::ptr::copy_nonoverlapping(chunk.as_ptr(), buf.add(offset), chunk.len());
            offset += chunk.len();
        }
        debug_assert_eq!(offset, total_len);

        if std::slice::from_raw_parts(buf, total_len).is_ascii() {
            return Ok(Bound::from_owned_ptr(py, scratch)
                .cast_into_unchecked()
                .unbind());
        }

        let errors = CString::new(errors)?;
        let decoded = pyo3::ffi::PyUnicode_DecodeUTF8(
            buf.cast::<std::os::raw::c_char>(),
            total_len as pyo3::ffi::Py_ssize_t,
            errors.as_ptr(),
        );
        pyo3::ffi::Py_DECREF(scratch);
        if decoded.is_null() {
            return Err(PyErr::fetch(py));
        }
        Ok(Bound::from_owned_ptr(py, decoded)
            .cast_into_unchecked()
            .unbind())
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
