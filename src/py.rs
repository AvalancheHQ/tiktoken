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
        // stream every token's bytes straight into the destination Python
        // `bytes` object in a single pass over the list.
        //
        // The previous shape of this path walked the tokens twice: once to
        // resolve each token to its byte slice into a `Vec<&[u8]>` and sum the
        // total length, then again to copy. That side table is 16 bytes per
        // token — for a 64 KiB document it is ~220 KiB written and read back,
        // more memory traffic than the decoded output itself, and the profile
        // showed the decode loop was memory-bound. Writing each token's bytes
        // into the output buffer as soon as it is resolved removes the table
        // (and the second traversal) entirely: the buffer is allocated from a
        // capacity estimate and trimmed to the exact length at the end.
        //
        // Note: unlike the non-`list` path, this holds the GIL for the whole
        // decode because it reads Python list items throughout. That is the
        // right trade single-threaded, but it does not release the GIL the way
        // `py.detach` does, so multi-threaded decoding on a GIL build won't
        // overlap here.
        if let Ok(list) = tokens.downcast::<PyList>() {
            // Tokens decode to ~4.6 bytes on average for natural-language text,
            // so this sizes the buffer generously enough that the common case
            // never has to grow, while staying within ~10% of the final size.
            let capacity = list.len().saturating_mul(5).saturating_add(64);
            let mut out = BytesWriter::new(py, capacity)?;
            // Items are read as borrowed pointers: the reference counts of the
            // token `int`s are left alone, which keeps their (scattered) object
            // headers clean instead of dirtying a cache line per token. The
            // critical section makes that sound on free-threaded builds by
            // locking the list against concurrent mutation; it compiles away on
            // GIL builds.
            let result: PyResult<()> =
                pyo3::sync::critical_section::with_critical_section(&list, || {
                    for index in 0..list.len() {
                        // SAFETY: `index` is in bounds, and the list cannot be
                        // mutated while we hold the critical section, so the
                        // borrowed item stays alive for the iteration.
                        let item = unsafe {
                            pyo3::ffi::PyList_GET_ITEM(
                                list.as_ptr(),
                                index as pyo3::ffi::Py_ssize_t,
                            )
                        };
                        let token = read_rank(py, item)?;
                        let token_bytes = self.token_bytes(token).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(format!(
                                "Invalid token for decoding: {token}"
                            ))
                        })?;
                        out.extend(py, token_bytes)?;
                    }
                    Ok(())
                });
            result?;
            return out.finish(py);
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

/// Copies one token's bytes to `dst`.
///
/// Tokens are short (a handful of bytes for natural-language text), and at that
/// size the call into `memcpy` costs more than the copy itself. Lengths up to 16
/// bytes are handled with a pair of overlapping fixed-width loads and stores,
/// which never touch a byte outside `src`/`dst[..src.len()]`.
///
/// # Safety
///
/// `dst` must be valid for writes of `src.len()` bytes and must not overlap
/// `src`.
#[inline]
unsafe fn copy_token_bytes(src: &[u8], dst: *mut u8) {
    let n = src.len();
    let src = src.as_ptr();
    unsafe {
        if n >= 8 {
            if n > 16 {
                std::ptr::copy_nonoverlapping(src, dst, n);
                return;
            }
            let head = src.cast::<u64>().read_unaligned();
            let tail = src.add(n - 8).cast::<u64>().read_unaligned();
            dst.cast::<u64>().write_unaligned(head);
            dst.add(n - 8).cast::<u64>().write_unaligned(tail);
        } else if n >= 4 {
            let head = src.cast::<u32>().read_unaligned();
            let tail = src.add(n - 4).cast::<u32>().read_unaligned();
            dst.cast::<u32>().write_unaligned(head);
            dst.add(n - 4).cast::<u32>().write_unaligned(tail);
        } else if n > 0 {
            // 1..=3 bytes: first, middle and last, which overlap for n < 3.
            *dst = *src;
            *dst.add(n / 2) = *src.add(n / 2);
            *dst.add(n - 1) = *src.add(n - 1);
        }
    }
}

/// An append-only writer over the buffer of a Python `bytes` object.
///
/// The object is allocated up front from a capacity estimate, filled in place,
/// and trimmed to the exact number of bytes written by `finish`. This lets a
/// decode loop copy each token's bytes directly into their final destination as
/// they are resolved, with no intermediate buffer or side table, and no second
/// traversal to compute the output length first.
struct BytesWriter {
    /// Owned strong reference to the `bytes` object being filled, or null once
    /// ownership has been handed over (`finish`) or released (resize failure).
    obj: *mut pyo3::ffi::PyObject,
    /// Interior buffer of `obj`. Re-read after every resize, which may move it.
    buf: *mut u8,
    len: usize,
    cap: usize,
}

impl BytesWriter {
    fn new(py: Python<'_>, cap: usize) -> PyResult<Self> {
        // SAFETY: we hold the GIL. A null data pointer asks CPython for an
        // uninitialised buffer of `cap` bytes, which we fill before anyone can
        // observe the object.
        let obj = unsafe {
            pyo3::ffi::PyBytes_FromStringAndSize(std::ptr::null(), cap as pyo3::ffi::Py_ssize_t)
        };
        if obj.is_null() {
            return Err(PyErr::fetch(py));
        }
        let buf = unsafe { pyo3::ffi::PyBytes_AS_STRING(obj) }
            .cast::<u8>()
            .cast_mut();
        Ok(Self {
            obj,
            buf,
            len: 0,
            cap,
        })
    }

    #[inline]
    fn extend(&mut self, py: Python<'_>, bytes: &[u8]) -> PyResult<()> {
        if self.len + bytes.len() > self.cap {
            self.grow(py, bytes.len())?;
        }
        // SAFETY: the branch above guarantees `bytes.len()` spare capacity at
        // `self.len`, and `bytes` borrows from the decoder, never from `buf`.
        unsafe { copy_token_bytes(bytes, self.buf.add(self.len)) };
        self.len += bytes.len();
        Ok(())
    }

    /// Grows the buffer so that at least `additional` more bytes fit.
    #[cold]
    fn grow(&mut self, py: Python<'_>, additional: usize) -> PyResult<()> {
        let cap = (self.len + additional).max(self.cap * 2);
        // SAFETY: we hold the only reference to `obj`, which is the contract of
        // `_PyBytes_Resize`. On failure it clears `obj` for us (the object is
        // deallocated), so we must not decrement its refcount afterwards.
        let ok =
            unsafe { pyo3::ffi::_PyBytes_Resize(&mut self.obj, cap as pyo3::ffi::Py_ssize_t) == 0 };
        if !ok {
            self.obj = std::ptr::null_mut();
            return Err(PyErr::fetch(py));
        }
        self.buf = unsafe { pyo3::ffi::PyBytes_AS_STRING(self.obj) }
            .cast::<u8>()
            .cast_mut();
        self.cap = cap;
        Ok(())
    }

    /// Trims the object to the bytes actually written and hands it over.
    fn finish(mut self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        if self.len != self.cap {
            // SAFETY: see `grow`; shrinking follows the same contract.
            let ok = unsafe {
                pyo3::ffi::_PyBytes_Resize(&mut self.obj, self.len as pyo3::ffi::Py_ssize_t) == 0
            };
            if !ok {
                self.obj = std::ptr::null_mut();
                return Err(PyErr::fetch(py));
            }
        }
        let obj = std::mem::replace(&mut self.obj, std::ptr::null_mut());
        // SAFETY: `obj` is a valid, fully initialised `bytes` object and we own
        // the reference we are transferring here.
        let bytes = unsafe { Bound::from_owned_ptr(py, obj).cast_into_unchecked::<PyBytes>() };
        Ok(bytes.unbind())
    }
}

impl Drop for BytesWriter {
    fn drop(&mut self) {
        if !self.obj.is_null() {
            // SAFETY: `BytesWriter` is only reachable while the GIL is held and
            // owns this reference.
            unsafe { pyo3::ffi::Py_DECREF(self.obj) };
        }
    }
}

/// Read a token id from a borrowed Python object pointer, reading a plain
/// in-range `int` with a direct CPython C-API call and otherwise falling back to
/// pyo3's `extract`.
#[inline]
fn read_rank(py: Python<'_>, item: *mut pyo3::ffi::PyObject) -> PyResult<Rank> {
    // SAFETY: `item` is a valid, non-null borrowed Python object for the
    // duration of the call, and the caller keeps it alive.
    let value = unsafe { pyo3::ffi::PyLong_AsUnsignedLong(item) };
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
    // SAFETY: as above; this takes its own reference for the duration of the
    // fallback conversion.
    let item = unsafe { Bound::from_borrowed_ptr(py, item) };
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
