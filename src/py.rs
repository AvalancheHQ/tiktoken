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
        // Fast path for the common case where `tokens` is a concrete `list`:
        // append every token's bytes straight into the destination Python
        // `bytes` object in a single streaming pass, growing the object when it
        // runs out of room and trimming it to the exact length at the end.
        //
        // The previous shape of this fast path had to know the total decoded
        // length before it could allocate the `bytes` object, which forced it to
        // walk the tokens once to resolve and total them, remembering a `&[u8]`
        // per token (16 bytes each — ~224 KB of scratch for a 64 KB document)
        // just to hand the slices to the copy pass. Streaming removes that
        // scratch buffer, and with it the memory traffic of writing it and
        // reading it back, without resolving any token twice.
        //
        // Note: unlike the non-`list` path, this holds the GIL for the whole
        // decode because it reads Python list items throughout. That is the
        // right trade single-threaded, but it does not release the GIL the way
        // `py.detach` does, so multi-threaded decoding on a GIL build won't
        // overlap here.
        if let Ok(list) = tokens.downcast::<PyList>() {
            // Tokens decode to about four bytes each on natural-language text,
            // so this sizes the buffer in one shot for text-like input; anything
            // denser grows geometrically, and the buffer is trimmed to the exact
            // length once at the end.
            let mut buf = BytesBuf::with_capacity(py, list.len().saturating_mul(5))?;
            for item in list.iter() {
                let token = read_rank(&item)?;
                let token_bytes = self.token_bytes(token).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "Invalid token for decoding: {token}"
                    ))
                })?;
                buf.extend_from_slice(token_bytes)?;
            }
            return buf.finish();
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

/// A Python `bytes` object being filled in place.
///
/// Owns a `bytes` object that nothing else references yet, so bytes can be
/// appended straight into it, the object grown with `_PyBytes_Resize` when it
/// runs out of room, and finally trimmed to the exact number of bytes written.
/// The caller therefore never has to know the total length up front, and so does
/// not have to remember anything per token. `Drop` releases the object if the
/// decode returns early (an invalid token, say).
///
/// This is the same shape as CPython 3.15's own `PyBytesWriter`, spelled out
/// here so it works on the Python versions tiktoken supports.
struct BytesBuf<'py> {
    py: Python<'py>,
    /// Owned, non-null `bytes` object with a refcount of exactly one.
    obj: *mut pyo3::ffi::PyObject,
    /// Start of `obj`'s writable buffer, cached to keep the append path free of
    /// Python calls. Refreshed whenever `obj` is reallocated.
    data: *mut u8,
    /// Bytes written so far; always `<= capacity`.
    len: usize,
    /// Allocated size of `obj`.
    capacity: usize,
}

impl<'py> BytesBuf<'py> {
    fn with_capacity(py: Python<'py>, capacity: usize) -> PyResult<Self> {
        // SAFETY: a null data pointer asks CPython for an uninitialised buffer
        // of the given size, which we then fill (and trim) ourselves.
        let obj = unsafe {
            pyo3::ffi::PyBytes_FromStringAndSize(
                std::ptr::null(),
                capacity as pyo3::ffi::Py_ssize_t,
            )
        };
        if obj.is_null() {
            return Err(PyErr::fetch(py));
        }
        // SAFETY: `obj` is a valid `bytes` object.
        let data = unsafe { pyo3::ffi::PyBytes_AS_STRING(obj).cast::<u8>().cast_mut() };
        Ok(Self {
            py,
            obj,
            data,
            len: 0,
            capacity,
        })
    }

    #[inline]
    fn extend_from_slice(&mut self, bytes: &[u8]) -> PyResult<()> {
        if bytes.len() > self.capacity - self.len {
            self.grow(self.len + bytes.len())?;
        }
        // SAFETY: `data` is valid for `capacity` bytes and the check above
        // guarantees `len + bytes.len() <= capacity`. The source cannot alias
        // the destination: it borrows the tokeniser's vocabulary, not the
        // freshly allocated `bytes` object.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(self.len), bytes.len());
        }
        self.len += bytes.len();
        Ok(())
    }

    #[cold]
    fn grow(&mut self, needed: usize) -> PyResult<()> {
        self.resize(needed.max(self.capacity + self.capacity / 2))
    }

    fn resize(&mut self, capacity: usize) -> PyResult<()> {
        // SAFETY: `obj` is a uniquely referenced `bytes` object, which is what
        // `_PyBytes_Resize` requires; it replaces the pointer on success and
        // clears it on failure.
        let res =
            unsafe { pyo3::ffi::_PyBytes_Resize(&mut self.obj, capacity as pyo3::ffi::Py_ssize_t) };
        if res != 0 || self.obj.is_null() {
            // `_PyBytes_Resize` already released the old object.
            self.obj = std::ptr::null_mut();
            self.data = std::ptr::null_mut();
            return Err(PyErr::fetch(self.py));
        }
        // SAFETY: `obj` is a valid `bytes` object.
        self.data = unsafe {
            pyo3::ffi::PyBytes_AS_STRING(self.obj)
                .cast::<u8>()
                .cast_mut()
        };
        self.capacity = capacity;
        Ok(())
    }

    /// Trims the object to the bytes actually written and hands over ownership.
    fn finish(mut self) -> PyResult<Py<PyBytes>> {
        if self.len != self.capacity {
            self.resize(self.len)?;
        }
        let obj = std::mem::replace(&mut self.obj, std::ptr::null_mut());
        // SAFETY: `obj` is an owned reference to a `bytes` object, and we no
        // longer touch it (`Drop` sees a null pointer).
        let bytes: Bound<'py, PyBytes> =
            unsafe { Bound::from_owned_ptr(self.py, obj).cast_into_unchecked() };
        Ok(bytes.unbind())
    }
}

impl Drop for BytesBuf<'_> {
    fn drop(&mut self) {
        // SAFETY: `obj` is either null or an owned reference we still hold.
        unsafe { pyo3::ffi::Py_XDECREF(self.obj) }
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
