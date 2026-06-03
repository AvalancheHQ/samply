use core::fmt::Write;
use libc::c_void;

struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        // Format into a fixed stack buffer — there is no allocator in a preload.
        // Messages are short; on overflow we just write the truncated prefix.
        let mut buf = heapless::String::<512>::new();
        let _ = writeln!(buf, "{}", record.args());
        unsafe {
            libc::write(
                libc::STDERR_FILENO,
                buf.as_ptr() as *const c_void,
                buf.len(),
            );
        }
    }

    fn flush(&self) {}
}

static LOGGER: StderrLogger = StderrLogger;

/// Install the stderr logger iff `SAMPLY_PRELOAD_DEBUG` is set. Idempotent
/// (a second `set_logger` simply fails and is ignored). Call once, early, from
/// the preload constructor; if never called the max level stays `Off`.
pub(crate) fn init() {
    let enabled = unsafe { !libc::getenv(c"SAMPLY_PRELOAD_DEBUG".as_ptr()).is_null() };
    if enabled && log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Trace);
    }
}
