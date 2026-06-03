//! The exec-path policy: decide what should actually be exec'd. Detects SIP
//! system binaries and `#!`-shebang scripts whose interpreter is one, and
//! redirects to an ad-hoc re-signed copy in the cache.

use core::ffi::CStr;
use core::ptr;
use libc::{c_char, c_void, mode_t};

use super::buffers::{CPathBuffer, ShebangArgv};
use super::sys::{codesign_adhoc, copy_file, count_argv, getenv};

/// Env var (set by samply) holding the writable cache root for re-signed copies.
/// When absent, the whole feature is inert.
const REDIRECT_DIR_ENV: &CStr = c"SAMPLY_SIP_REDIRECT_DIR";

const SHEBANG_READ_SZ: usize = 256;

/// Returns the path that should actually be exec'd.  If the target is a script
/// with a SIP interpreter in its shebang, redirects the *interpreter* instead
/// and populates `shebang_argv` with the rebuilt argv.
///
/// On any failure we fall back to the original `path` — the worst case is the
/// pre-existing behaviour (DYLD stripped), never a failed exec.
pub(super) unsafe fn redirect(
    path: *const c_char,
    argv: *const *const c_char,
    out: &mut CPathBuffer,
    shebang_argv: &mut ShebangArgv,
) -> *const c_char {
    if path.is_null() {
        return path;
    }

    let path = CStr::from_ptr(path);
    let Some(cache_root) = getenv(REDIRECT_DIR_ENV) else {
        log::debug!("sip_redirect: pass-through (feature disabled)");
        return path.as_ptr();
    };

    // Direct SIP binary redirect?
    if is_redirectable(path) {
        if ensure_resigned_copy(cache_root, path, out) {
            log::debug!("sip_redirect: redirected: {path:?}");
            return out.as_ptr();
        }
        log::debug!("sip_redirect: pass-through (resign failed): {path:?}");
        return path.as_ptr();
    }

    // Not a SIP binary — try shebang redirect
    if try_shebang_redirect(cache_root, path, argv, out, shebang_argv) {
        log::debug!("sip_redirect: shebang redirect: {path:?}");
        return out.as_ptr();
    }

    log::debug!("sip_redirect: pass-through (not redirectable): {path:?}");
    path.as_ptr()
}

/// If `path` is a regular file with a `#!` shebang whose interpreter is a SIP
/// binary, resign the interpreter into `out` and populate `shebang_argv`.
unsafe fn try_shebang_redirect(
    cache_root: &CStr,
    path: &CStr,
    argv: *const *const c_char,
    out: &mut CPathBuffer,
    shebang_argv: &mut ShebangArgv,
) -> bool {
    // fstat: only regular files can be scripts (handles relative paths via CWD)
    let mut st: libc::stat = core::mem::zeroed();
    if libc::stat(path.as_ptr(), &mut st) != 0 {
        return false;
    }
    if (st.st_mode & libc::S_IFMT) != libc::S_IFREG {
        return false;
    }

    // Read first bytes looking for shebang
    let mut read_buf = [0u8; SHEBANG_READ_SZ];
    let fd = libc::open(path.as_ptr(), libc::O_RDONLY);
    if fd < 0 {
        return false;
    }
    let nread = libc::read(fd, read_buf.as_mut_ptr() as *mut c_void, read_buf.len());
    libc::close(fd);
    if nread < 2 || read_buf[0] != b'#' || read_buf[1] != b'!' {
        return false;
    }

    // Parse "#!  <interpreter> [<arg>] \n"
    let line = match read_buf[..nread as usize].iter().position(|&b| b == b'\n') {
        Some(pos) => &read_buf[..pos],
        None => return false, // truncated
    };

    // Skip "#!" and leading whitespace
    let mut start = 2;
    while start < line.len() && line[start] == b' ' {
        start += 1;
    }
    if start >= line.len() {
        return false;
    }

    // Find end of interpreter path
    let interp_end = line[start..]
        .iter()
        .position(|&b| b == b' ')
        .map(|p| start + p)
        .unwrap_or(line.len());

    let interp_bytes = &line[start..interp_end];
    if interp_bytes.is_empty() {
        return false;
    }

    // Nul-terminate the interpreter path on the stack
    let mut interp_cstr_buf = [0u8; libc::PATH_MAX as usize];
    let ilen = interp_bytes.len().min(libc::PATH_MAX as usize - 1);
    interp_cstr_buf[..ilen].copy_from_slice(&interp_bytes[..ilen]);

    let interp_cstr = CStr::from_bytes_with_nul_unchecked(&interp_cstr_buf[..=ilen]);
    if !is_redirectable(interp_cstr) {
        log::debug!("sip_redirect: shebang interpreter not redirectable: {interp_cstr:?}");
        return false;
    }

    // Redirect (resign) the interpreter into `out`
    if !ensure_resigned_copy(cache_root, interp_cstr, out) {
        log::debug!("sip_redirect: shebang resign failed");
        return false;
    }

    // Count original argc
    let argc = count_argv(argv);

    // Build new argv:
    //   [resigned_interpreter, shebang_arg?, original_path, original_argv[1..]]
    shebang_argv.push(out.as_ptr());

    // Optional shebang argument (e.g., "node" in "#!/usr/bin/env node"). Its
    // bytes are copied into storage owned by `shebang_argv`, which outlives this
    // function — a local buffer would dangle on return and corrupt the argv.
    if interp_end < line.len() {
        let arg_bytes = &line[interp_end + 1..];
        // Trim at the next whitespace (only a single interpreter arg is supported).
        let arg_end = arg_bytes
            .iter()
            .position(|&b| b == b' ')
            .unwrap_or(arg_bytes.len());
        let arg = &arg_bytes[..arg_end];
        if !arg.is_empty() && !shebang_argv.push_arg_bytes(arg) {
            log::debug!("sip_redirect: shebang arg overflow");
            return false;
        }
    }

    shebang_argv.push(path.as_ptr());

    for i in 1..argc {
        let a = *argv.add(i);
        if a.is_null() {
            break;
        }
        if !shebang_argv.push(a) {
            log::debug!("sip_redirect: shebang argv overflow");
            return false;
        }
    }

    shebang_argv.push(ptr::null());
    true
}

/// Check whether `path` is a SIP-protected binary that is safe to ad-hoc re-sign
unsafe fn is_redirectable(path: &CStr) -> bool {
    // Never the signer itself — we shell out to it to do the re-signing.
    if path.to_bytes() == b"/usr/bin/codesign" {
        return false;
    }

    let mut st: libc::stat = core::mem::zeroed();
    if libc::stat(path.as_ptr(), &mut st) != 0 {
        return false;
    }

    // Check if it's a SIP-protected system binary.
    const SF_RESTRICTED: u32 = 0x0008_0000; // from <sys/stat.h>`
    if st.st_flags & SF_RESTRICTED == 0 {
        return false;
    }

    // Never setuid/setgid: re-signing drops the bit (and entitlements) and would
    // break privileged tools (`sudo`, `login`, …)
    if (st.st_mode & (libc::S_ISUID | libc::S_ISGID) as mode_t) != 0 {
        return false;
    }

    true
}

/// Ensure `<cache_root><path>` exists as an ad-hoc re-signed copy of `path`,
/// building it on first use. Writes that destination path into `out` and returns
/// whether it is usable.
unsafe fn ensure_resigned_copy(cache_root: &CStr, path: &CStr, out: &mut CPathBuffer) -> bool {
    // dst = cache_root + path  (path starts with '/', so this mirrors the tree)
    if out.set_concat(cache_root, path).is_none() {
        return false;
    }

    // Already built and not older than the source? Reuse it.
    let mut dst_st: libc::stat = core::mem::zeroed();
    let mut src_st: libc::stat = core::mem::zeroed();
    if libc::stat(out.as_ptr(), &mut dst_st) == 0
        && libc::stat(path.as_ptr(), &mut src_st) == 0
        && dst_st.st_mtime >= src_st.st_mtime
    {
        log::debug!("sip_redirect: reusing cached copy");
        return true;
    }

    out.make_parent_dirs();

    // Build to a unique temp then atomically rename, so concurrent execs never
    // observe a half-written / unsigned copy.
    let mut tmp = CPathBuffer::new();
    if !tmp.set_tmp_path(out) {
        return false;
    }

    log::debug!("sip_redirect: copying and ad-hoc signing");
    if copy_file(path, tmp.as_c_str())
        && codesign_adhoc(tmp.as_c_str())
        && libc::rename(tmp.as_ptr(), out.as_ptr()) == 0
    {
        true
    } else {
        log::debug!("sip_redirect: resign/rename failed");
        libc::unlink(tmp.as_ptr());
        false
    }
}
