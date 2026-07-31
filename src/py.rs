use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};

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
            //
            // `compact_ints` is hoisted out of the loop: it is a one-off check
            // that the interpreter lays its `int` objects out the way
            // `read_compact_rank` expects, so the per-token path is a plain
            // predictable branch.
            let compact_ints = COMPACT_INT_LAYOUT_OK.load(Ordering::Relaxed);
            let mut slices: Vec<&[u8]> = Vec::with_capacity(list.len());
            let mut total_len = 0usize;
            for item in list.iter() {
                // SAFETY: `item` is a valid, non-null borrowed Python object and
                // we hold the GIL. The layout the reader relies on was verified
                // against this interpreter at import time.
                let token = match compact_ints
                    .then(|| unsafe { read_compact_rank(item.as_ptr()) })
                    .flatten()
                {
                    Some(rank) => rank,
                    None => read_rank(&item)?,
                };
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

/// Whether this interpreter stores its `int` objects the way
/// [`read_compact_rank`] reads them. Verified once at import time; until then
/// (and forever, on an interpreter whose layout we do not recognise) the decode
/// path uses the C-API reader only.
static COMPACT_INT_LAYOUT_OK: AtomicBool = AtomicBool::new(false);

/// Prefix of CPython 3.12+'s `PyLongObject`: the object header, then the
/// `_PyLongValue` tag and the first digit.
///
/// Only the prefix is declared, which is all the reader touches. `ob_base` is
/// pyo3's `PyObject`, so the header matches whatever build (free-threaded,
/// trace-refs, ...) the extension is compiled for.
#[repr(C)]
struct CompactPyLong {
    ob_base: pyo3::ffi::PyObject,
    lv_tag: usize,
    digit0: u32,
}

/// `lv_tag` packs the digit count in the bits above the 3 low sign bits, with
/// `SIGN_ZERO == 1` and `SIGN_NEGATIVE == 2`. So a zero is tagged `1` and a
/// single-digit non-negative `int` is tagged `1 << 3`. Every other tag (more
/// digits, or a negative sign) routes to the C-API reader.
const LONG_TAG_ZERO: usize = 1;
const LONG_TAG_ONE_DIGIT: usize = 8;

/// Read a non-negative single-digit `int` straight out of the object, without
/// calling into libpython.
///
/// Returns `None` for anything that is not an exact, non-negative, single-digit
/// `int`; the caller then falls back to [`read_rank`], so behaviour (including
/// error types) is unchanged.
///
/// # Safety
///
/// `item` must be a valid, non-null Python object and the GIL must be held.
#[inline]
unsafe fn read_compact_rank(item: *mut pyo3::ffi::PyObject) -> Option<Rank> {
    unsafe {
        if pyo3::ffi::PyLong_CheckExact(item) == 0 {
            return None;
        }
        let long = item.cast::<CompactPyLong>();
        match (*long).lv_tag {
            LONG_TAG_ZERO => Some(0),
            LONG_TAG_ONE_DIGIT => Some((*long).digit0),
            _ => None,
        }
    }
}

/// Check [`read_compact_rank`] against the running interpreter before the
/// decode path is allowed to use it.
///
/// Every value that must be read inline is round-tripped through a real
/// `int` object, and values that must be refused (negative, multi-digit) are
/// checked to be refused. An interpreter with a different `int` layout — an
/// older CPython, a 15-bit-digit build, PyPy — fails this and keeps using the
/// C-API reader.
fn compact_int_layout_is_valid() -> bool {
    for value in [
        0u32,
        1,
        2,
        255,
        256,
        65_535,
        100_000,
        200_000,
        (1 << 30) - 1,
    ] {
        // SAFETY: we hold the GIL (called from module initialisation).
        let obj = unsafe { pyo3::ffi::PyLong_FromUnsignedLong(value.into()) };
        if obj.is_null() {
            unsafe { pyo3::ffi::PyErr_Clear() };
            return false;
        }
        let read = unsafe { read_compact_rank(obj) };
        unsafe { pyo3::ffi::Py_DECREF(obj) };
        if read != Some(value) {
            return false;
        }
    }
    for value in [-1i64, -(1 << 40), 1 << 40] {
        // SAFETY: we hold the GIL (called from module initialisation).
        let obj = unsafe { pyo3::ffi::PyLong_FromLongLong(value) };
        if obj.is_null() {
            unsafe { pyo3::ffi::PyErr_Clear() };
            return false;
        }
        let read = unsafe { read_compact_rank(obj) };
        unsafe { pyo3::ffi::Py_DECREF(obj) };
        if read.is_some() {
            return false;
        }
    }
    true
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
    COMPACT_INT_LAYOUT_OK.store(compact_int_layout_is_valid(), Ordering::Relaxed);
    m.add_class::<CoreBPE>()?;
    Ok(())
}
