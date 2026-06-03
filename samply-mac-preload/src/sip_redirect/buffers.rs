//! Stack-allocated, caller-owned buffers used by the hooks: a nul-terminated C
//! path builder and a rebuilt-argv builder for shebang redirects. Every pointer
//! these hand to `exec`/`posix_spawn` must outlive the call, so they live in the
//! hook's own frame.

use core::ffi::CStr;
use heapless::Vec;
use libc::c_char;

pub(super) struct CPathBuffer {
    bytes: Vec<u8, { libc::PATH_MAX as usize }>,
}

impl CPathBuffer {
    pub(super) fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    pub(super) fn as_ptr(&self) -> *const c_char {
        self.bytes.as_ptr() as *const c_char
    }

    pub(super) fn as_c_str(&self) -> &CStr {
        unsafe { CStr::from_bytes_with_nul_unchecked(&self.bytes) }
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len()]
    }

    fn len(&self) -> usize {
        self.bytes.len().saturating_sub(1)
    }

    pub(super) fn set_concat(&mut self, a: &CStr, b: &CStr) -> Option<()> {
        self.bytes.clear();
        self.bytes.extend_from_slice(a.to_bytes()).ok()?;
        self.bytes.extend_from_slice(b.to_bytes()).ok()?;
        self.bytes.push(0).ok()?;
        Some(())
    }

    /// `<dst>.<pid>.tmp` for an atomic build.
    pub(super) fn set_tmp_path(&mut self, dst: &CPathBuffer) -> bool {
        let pid = unsafe { libc::getpid() };
        let mut itoa_buf = itoa::Buffer::new();
        let pid_str = itoa_buf.format(pid as u32);

        self.bytes.clear();
        self.bytes.extend_from_slice(dst.as_bytes()).is_ok()
            && self.bytes.push(b'.').is_ok()
            && self.bytes.extend_from_slice(pid_str.as_bytes()).is_ok()
            && self.bytes.extend_from_slice(b".tmp").is_ok()
            && self.bytes.push(0).is_ok()
    }

    /// `mkdir -p` on the parent directories of the nul-terminated path.
    pub(super) unsafe fn make_parent_dirs(&mut self) {
        // Walk each '/' (skip index 0), temporarily terminate, mkdir, restore.
        let mut i = 1;
        while i < self.len() {
            if self.bytes[i] == b'/' {
                self.bytes[i] = 0;
                libc::mkdir(self.as_ptr(), 0o755);
                self.bytes[i] = b'/';
            }
            i += 1;
        }
    }
}

/// Max slots in a rebuilt argv for shebang redirects.
const MAX_SHEBANG_ARGS: usize = 64;

/// Max bytes for the inline shebang-argument backing store (e.g. "node").
const SHEBANG_ARG_CAP: usize = 128;

/// Stack-allocated argv for shebang redirects. The caller pushes the trailing
/// null explicitly, so `as_slice` returns a ready-to-use, null-terminated argv.
pub(super) struct ShebangArgv {
    ptrs: Vec<*const c_char, MAX_SHEBANG_ARGS>,
    /// Backing storage for the optional shebang interpreter argument. This must
    /// live as long as the rebuilt argv (i.e. in the caller's frame, alongside
    /// `ptrs`), because one of `ptrs` points into it. A local buffer in
    /// `try_shebang_redirect` would dangle on return and corrupt the argv.
    arg_buf: [u8; SHEBANG_ARG_CAP],
}

impl ShebangArgv {
    pub(super) fn new() -> Self {
        Self {
            ptrs: Vec::new(),
            arg_buf: [0; SHEBANG_ARG_CAP],
        }
    }

    pub(super) fn push(&mut self, p: *const c_char) -> bool {
        self.ptrs.push(p).is_ok()
    }

    /// Copy `bytes` into the owned `arg_buf` (nul-terminating) and push a
    /// pointer to it. Returns false on overflow of either store.
    pub(super) fn push_arg_bytes(&mut self, bytes: &[u8]) -> bool {
        let n = bytes.len().min(SHEBANG_ARG_CAP - 1);
        self.arg_buf[..n].copy_from_slice(&bytes[..n]);
        self.arg_buf[n] = 0;
        self.push(self.arg_buf.as_ptr() as *const c_char)
    }

    pub(super) fn as_slice(&self) -> Option<&[*const c_char]> {
        if self.ptrs.is_empty() {
            None
        } else {
            Some(self.ptrs.as_slice())
        }
    }
}
