//! Reconstruct V8 TurboFan/Maglev inline frames from the V8 code log, keyed by
//! JIT code address + size.
//!
//! V8's jitdump records only the *innermost* inlined source position per PC, so
//! a sampling profiler collapses an inlined chain `root -> a -> b -> c` down to
//! just `root`. The full inline tree is instead present in the code
//! log's `code-source-info` records (`inlining_positions` + `inlined_functions`).
//! Both use the same runtime code addresses, so joining by address expands a
//! single jitdump frame into the real inline stack.
//!
//! A code address is not unique over a process's lifetime: V8 frees code and
//! reuses the address for another object. We disambiguate by also keying on the
//! code object's byte size, which both the jitdump (`CODE_LOAD`) and the log
//! (`code-creation`) report identically and which is available at lookup time.

use std::collections::HashMap;
use std::path::Path;

/// One reconstructed inlined frame.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct V8Frame {
    /// Display name in V8's jitdump style: `JS:<marker><fn> <file>:<line>:<col>`
    /// (anonymous functions have an empty `<fn>`, e.g. `JS:* /path:17:32`). This
    /// matches how the jitdump names the standalone (non-inlined) code object
    /// for the same function, so inlined and non-inlined occurrences coalesce.
    pub name: String,
    pub file: String,
    pub line: u32,
    pub col: u32,
}

/// One entry of the source-position table, mapping a code offset to a script
/// position and the inlining node (if any) it belongs to.
#[derive(Debug, Clone, Copy)]
struct SourcePosition {
    code_offset: u32,
    #[allow(dead_code)]
    script_offset: u32,
    inlining_id: Option<u32>,
}

/// One node of the inlining tree. `function_id` indexes `inlined_functions`;
/// it is absent for a node that has no resolved function (V8 logs an empty id).
#[derive(Debug, Clone, Copy)]
struct InliningNode {
    function_id: Option<u32>,
    #[allow(dead_code)]
    call_site_script_offset: u32,
    parent_inlining_id: Option<u32>,
}

#[derive(Debug, Default)]
struct CodeInline {
    /// Source-position table, sorted ascending by `code_offset`.
    source_positions: Vec<SourcePosition>,
    /// `code-source-info` inlining tree, indexed by inlining id.
    inlining_tree: Vec<InliningNode>,
    /// Resolved inlined functions, indexed by `function_id`, already formatted
    /// with this code object's tier marker.
    inlined_functions: Vec<V8Frame>,
}

/// Inline maps for every optimized JS code object that has inlining, keyed by
/// the code's runtime start address (== jitdump `CODE_LOAD` address) and byte
/// size (== jitdump `CODE_LOAD` size), so addresses reused over the run don't
/// collide.
#[derive(Debug, Default)]
pub struct V8InlineMap {
    by_key: HashMap<(u64, u32), CodeInline>,
}

impl V8InlineMap {
    pub fn from_log_file(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let map = Self::from_log_str(&text);
        if map.by_key.is_empty() {
            None
        } else {
            Some(map)
        }
    }

    pub fn from_log_str(text: &str) -> Self {
        // Pass 1: build sfi_addr -> (function, location) from `code-creation`
        // records (one is emitted per tier; any resolves the name). A function
        // can be inlined into code that is logged before the function's own
        // standalone code-creation, so we need the full map before pass 2.
        let mut sfi_fn: HashMap<u64, (String, String)> = HashMap::new();
        for line in text.lines() {
            let Some(rest) = line.strip_prefix("code-creation,") else {
                continue;
            };
            // <kind>,<tag>,<ts>,<addr>,<size>,<name>,<sfi>,<opt?>
            let p: Vec<&str> = rest.splitn(8, ',').collect();
            if p.len() < 7 {
                continue;
            }
            let Some(sfi) = parse_hex(p[6]) else {
                continue;
            };
            sfi_fn.entry(sfi).or_insert_with(|| parse_name(p[5]));
        }

        // Pass 2 (in order): parse `code-source-info` records that carry an
        // inlining tree, pairing each with its code object's size + tier marker.
        // V8 emits `code-creation` immediately followed by the matching
        // `code-source-info` for the same address, so the last code-creation
        // seen at an address identifies the object the code-source-info belongs
        // to (and supplies its size, which keys the map, and its tier marker,
        // which formats the frame names).
        let mut by_key = HashMap::new();
        let mut live_at_addr: HashMap<u64, (u32, String)> = HashMap::new();
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("code-creation,") {
                let p: Vec<&str> = rest.splitn(8, ',').collect();
                if p.len() < 7 {
                    continue;
                }
                let (Some(addr), Ok(size)) = (parse_hex(p[3]), p[4].parse::<u32>()) else {
                    continue;
                };
                let marker = p.get(7).copied().unwrap_or("").to_string();
                live_at_addr.insert(addr, (size, marker));
            } else if let Some(rest) = line.strip_prefix("code-source-info,") {
                // <addr>,<script>,<start>,<end>,<positions>,<inlining_positions>,<inlined_functions>
                let p: Vec<&str> = rest.splitn(7, ',').collect();
                if p.len() < 7 || p[5].is_empty() {
                    continue; // no inlining tree -> leave to the plain jitdump path
                }
                let Some(addr) = parse_hex(p[0]) else {
                    continue;
                };
                let Some((size, marker)) = live_at_addr.get(&addr) else {
                    continue; // code-source-info without a preceding code-creation
                };
                let mut source_positions = parse_positions(p[4]);
                source_positions.sort_by_key(|sp| sp.code_offset);
                let inlining_tree = parse_inlining_positions(p[5]);
                let inlined_functions = parse_inlined_functions(p[6])
                    .into_iter()
                    .map(|sfi| {
                        let (func, loc) = sfi_fn.get(&sfi).cloned().unwrap_or_default();
                        make_frame(marker, &func, &loc)
                    })
                    .collect();
                by_key.insert(
                    (addr, *size),
                    CodeInline {
                        source_positions,
                        inlining_tree,
                        inlined_functions,
                    },
                );
            }
        }
        Self { by_key }
    }

    /// Reconstruct the inlined frames (innermost first) for a PC at `pc_offset`
    /// within the code object at (`code_addr`, `code_size`). Returns `None` when
    /// there is no genuine inlining at that PC; the code object's own frame is
    /// supplied by the caller (the jitdump symbol), so it is *not* included.
    pub fn lookup(&self, code_addr: u64, code_size: u32, pc_offset: u64) -> Option<Vec<V8Frame>> {
        let code_inlines = self.by_key.get(&(code_addr, code_size))?;
        // Source-position table: an entry applies from its code_offset until the
        // next one, so take the last entry with code_offset <= pc_offset.
        let mut inlining_id = None;
        for sp in &code_inlines.source_positions {
            if u64::from(sp.code_offset) <= pc_offset {
                inlining_id = sp.inlining_id;
            } else {
                break;
            }
        }
        let mut frames = Vec::new();
        // Walk the inlining tree from the innermost node up to the root, the same
        // way V8 reconstructs an inlined call stack in
        // `SourcePosition::InliningStack` (src/codegen/source-position.cc).
        let mut guard = 0;
        while let Some(id) = inlining_id {
            let node = code_inlines.inlining_tree.get(id as usize)?;
            if let Some(function_id) = node.function_id {
                if let Some(frame) = code_inlines.inlined_functions.get(function_id as usize) {
                    frames.push(frame.clone());
                }
            }
            inlining_id = node.parent_inlining_id;
            guard += 1;
            if guard > 64 {
                break; // defensive against a malformed cycle
            }
        }
        if frames.is_empty() {
            return None;
        }
        Some(frames)
    }
}

/// Reduce a V8 `code-creation` tier marker to the bare tier character used by
/// the jitdump / perf-map naming convention.
///
/// `code-creation` records carry the marker from `CodeKindToMarker`, which
/// decorates the optimized tiers with a trailing context-specialization quote
/// (`+'`, `*'`) and an OSR prefix (`o+`, `o*`). The jitdump and perf map name
/// the same code object with the plain tier marker (`~`, `^`, `+`, `*`), so a
/// reconstructed inlined frame must drop those decorations to coalesce with the
/// standalone frame for the same function.
fn canonical_tier_marker(marker: &str) -> &str {
    let marker = marker.strip_prefix('o').unwrap_or(marker);
    marker.strip_suffix('\'').unwrap_or(marker)
}

/// Build a frame whose display name matches the jitdump's: `JS:<marker><fn>`
/// for an anonymous function with no location, otherwise
/// `JS:<marker><fn> <file>:<line>:<col>` (V8 puts a single space before the
/// location and no space between the tier marker and the name).
fn make_frame(marker: &str, func: &str, loc: &str) -> V8Frame {
    let marker = canonical_tier_marker(marker);
    let (file, line, col) = parse_location(loc);
    let name = if loc.is_empty() {
        format!("JS:{marker}{func}")
    } else {
        format!("JS:{marker}{func} {loc}")
    };
    V8Frame {
        name,
        file,
        line,
        col,
    }
}

fn parse_hex(s: &str) -> Option<u64> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(s, 16).ok()
}

/// `"getNoiseRatio /path/x.ts:28:23"` -> `("getNoiseRatio", "/path/x.ts:28:23")`;
/// an anonymous function `" /path/x.ts:17:32"` -> `("", "/path/x.ts:17:32")`.
fn parse_name(name: &str) -> (String, String) {
    let (func, loc) = name.split_once(' ').unwrap_or((name, ""));
    (func.to_string(), loc.to_string())
}

/// `"/path/x.ts:28:23"` -> `("/path/x.ts", 28, 23)`.
fn parse_location(loc: &str) -> (String, u32, u32) {
    let mut it = loc.rsplitn(3, ':');
    let col = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let line = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let file = it.next().unwrap_or("").to_string();
    (file, line, col)
}

fn read_u32(b: &[u8], i: &mut usize) -> u32 {
    let mut n: u32 = 0;
    while *i < b.len() && b[*i].is_ascii_digit() {
        n = n * 10 + u32::from(b[*i] - b'0');
        *i += 1;
    }
    n
}

/// `"C20O1284C40O914I0..."` -> source positions `[{20,1284,None},{40,914,Some(0)},...]`.
fn parse_positions(s: &str) -> Vec<SourcePosition> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        if b[i] != b'C' {
            break;
        }
        i += 1;
        let code_offset = read_u32(b, &mut i);
        if i >= b.len() || b[i] != b'O' {
            break;
        }
        i += 1;
        let script_offset = read_u32(b, &mut i);
        let mut inlining_id = None;
        if i < b.len() && b[i] == b'I' {
            i += 1;
            inlining_id = Some(read_u32(b, &mut i));
        }
        out.push(SourcePosition {
            code_offset,
            script_offset,
            inlining_id,
        });
    }
    out
}

/// `"F0O1298F1O1046I0"` -> inlining nodes `[{Some(0),1298,None},{Some(1),1046,Some(0)}]`.
/// A missing number after `F` means the node has no resolved function, so
/// `function_id` is `None`.
fn parse_inlining_positions(s: &str) -> Vec<InliningNode> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        if b[i] != b'F' {
            break;
        }
        i += 1;
        let function_id = if i < b.len() && b[i].is_ascii_digit() {
            Some(read_u32(b, &mut i))
        } else {
            None
        };
        if i >= b.len() || b[i] != b'O' {
            break;
        }
        i += 1;
        let call_site_script_offset = read_u32(b, &mut i);
        let mut parent_inlining_id = None;
        if i < b.len() && b[i] == b'I' {
            i += 1;
            parent_inlining_id = Some(read_u32(b, &mut i));
        }
        out.push(InliningNode {
            function_id,
            call_site_script_offset,
            parent_inlining_id,
        });
    }
    out
}

/// `"S0x3fb3a2e510S0x3fb3a2e4c8"` -> `[0x3fb3a2e510, 0x3fb3a2e4c8]`.
fn parse_inlined_functions(s: &str) -> Vec<u64> {
    s.split('S')
        .filter(|t| !t.is_empty())
        .filter_map(parse_hex)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real records captured from a V8 inlining repro (custom node 24), plus a
    // second object at the *same address* with a different size to exercise the
    // address-reuse disambiguation.
    const LOG: &str = "\
code-creation,JS,9,337804,0x2db9f56b98,17,benchFn /tmp/repro_inline/repro.js:37:17,0x3fb3a2e558,~
code-creation,JS,9,337915,0x2db9f56c50,140,getNoiseRatio /tmp/repro_inline/repro.js:26:23,0x3fb3a2e510,~
code-creation,JS,9,338046,0x2db9f56db0,186,rollingMean /tmp/repro_inline/repro.js:10:21,0x3fb3a2e4c8,~
code-creation,JS,12,344459,0xffffc73cb7a0,1716,benchFn /tmp/repro_inline/repro.js:37:17,0x3fb3a2e558,*'
code-source-info,0xffffc73cb7a0,85,1284,1324,C20O1284C40O914I0C172O358I1C228O975I0,F0O1298F1O1046I0,S0x3fb3a2e510S0x3fb3a2e4c8
code-creation,JS,12,999999,0xffffc73cb7a0,512,other /tmp/repro_inline/repro.js:99:1,0x3fb3a2dead,*'
code-source-info,0xffffc73cb7a0,85,1,2,C0O1I0,F0O1,S0x3fb3a2e558";

    fn names(frames: Vec<V8Frame>) -> Vec<String> {
        frames.into_iter().map(|f| f.name).collect()
    }

    #[test]
    fn reconstructs_full_inline_chain() {
        let map = V8InlineMap::from_log_str(LOG);
        let addr = 0xffffc73cb7a0;
        let size = 1716;

        // PC at code_offset 172 is inside inlined rollingMean (I1): the chain is
        // rollingMean -> getNoiseRatio (the code object's own benchFn frame is
        // added by the caller, so it is not returned here).
        // The inlined frames carry the *outer* code object's tier marker, which
        // V8 logs as the context-specialized `*'` but is normalized to the bare
        // jitdump tier marker `*` so it coalesces with the standalone frame.
        assert_eq!(
            names(map.lookup(addr, size, 180).unwrap()),
            vec![
                "JS:*rollingMean /tmp/repro_inline/repro.js:10:21",
                "JS:*getNoiseRatio /tmp/repro_inline/repro.js:26:23",
            ]
        );
        // PC at code_offset 40 is inside inlined getNoiseRatio (I0).
        assert_eq!(
            names(map.lookup(addr, size, 100).unwrap()),
            vec!["JS:*getNoiseRatio /tmp/repro_inline/repro.js:26:23"]
        );
        // PC at code_offset 20 is in benchFn itself -> no inlined frames.
        assert_eq!(map.lookup(addr, size, 30), None);

        // Frame metadata is resolved from the sfi map.
        let leaf = &map.lookup(addr, size, 180).unwrap()[0];
        assert_eq!(leaf.file, "/tmp/repro_inline/repro.js");
        assert_eq!(leaf.line, 10);
        assert_eq!(leaf.col, 21);
    }

    #[test]
    fn address_reuse_is_disambiguated_by_size() {
        let map = V8InlineMap::from_log_str(LOG);
        let addr = 0xffffc73cb7a0;
        // The size-512 object reused the address; its only inlined function is
        // benchFn. Looking it up by its own size must not return the size-1716
        // object's chain.
        assert_eq!(
            names(map.lookup(addr, 512, 5).unwrap()),
            vec!["JS:*benchFn /tmp/repro_inline/repro.js:37:17"]
        );
        // The size-1716 object is still intact.
        assert_eq!(
            names(map.lookup(addr, 1716, 100).unwrap()),
            vec!["JS:*getNoiseRatio /tmp/repro_inline/repro.js:26:23"]
        );
    }

    #[test]
    fn anonymous_functions_get_a_location_based_name() {
        // Anonymous inlined function (empty name, valid location) must not yield
        // an empty frame name. The tier marker is normalized to the bare `*`.
        let f = make_frame("*'", "", "/path/x.ts:17:32");
        assert_eq!(f.name, "JS:* /path/x.ts:17:32");
        assert_eq!((f.file.as_str(), f.line, f.col), ("/path/x.ts", 17, 32));
    }

    #[test]
    fn tier_marker_is_normalized_to_jitdump_convention() {
        // Bare tier markers pass through unchanged.
        assert_eq!(canonical_tier_marker("~"), "~");
        assert_eq!(canonical_tier_marker("^"), "^");
        assert_eq!(canonical_tier_marker("+"), "+");
        assert_eq!(canonical_tier_marker("*"), "*");
        // Context-specialized optimized markers drop the trailing quote.
        assert_eq!(canonical_tier_marker("+'"), "+");
        assert_eq!(canonical_tier_marker("*'"), "*");
        // OSR-entry optimized markers drop the leading `o`.
        assert_eq!(canonical_tier_marker("o+"), "+");
        assert_eq!(canonical_tier_marker("o*"), "*");
    }
}
