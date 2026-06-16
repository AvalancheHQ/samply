use std::collections::BTreeMap;

/// Keeps track of mapped libraries in an address space. Stores a value
/// for each mapping, and allows efficient lookup of that value based on
/// an address.
///
/// A "library" here is a loose term; it could be a normal shared library,
/// or the main binary, but it could also be a synthetic library for JIT
/// code. For normal libraries, there's usually just one mapping per library.
/// For JIT code, you could have many small mappings, one per JIT function,
/// all pointing to the synthetic JIT "library".
#[derive(Debug, Clone)]
pub struct LibMappings<T> {
    /// A BTreeMap of non-overlapping Mappings. The key is the start_avma of the mapping.
    ///
    /// When a new mapping is added, the overlapping part of any existing mapping
    /// is carved away, but its non-overlapping remainder is kept.
    map: BTreeMap<u64, Mapping<T>>,
}

impl<T> Default for LibMappings<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LibMappings<T> {
    /// Creates a new empty instance.
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    /// Add a mapping to this address space.
    ///
    /// The new mapping reflects the current state of the address range it
    /// covers, so it takes precedence over whatever was mapped there before. But
    /// rather than dropping every existing mapping it touches, we only carve out
    /// the overlapping part and keep each existing mapping's non-overlapping
    /// remainder:
    ///
    /// - a mapping reaching into the new range from the left keeps its head
    ///   `[old_start, start_avma)`;
    /// - a mapping reaching out of the new range to the right keeps its tail
    ///   `[end_avma, old_end)` (its relative address is shifted accordingly);
    /// - a mapping that fully contains the new range keeps both a head and a
    ///   tail piece on either side of it;
    /// - a mapping fully inside the new range is dropped.
    ///
    /// This matters when a small mapping lands inside a larger one — e.g. the
    /// kernel placing `[vdso]`/`[vvar]` in the gap of an interpreter's
    /// over-long executable mapping. Dropping the larger mapping wholesale would
    /// lose symbolication for everything else it covered; keeping the remainder
    /// preserves it.
    ///
    /// `start_avma` and `end_avma` describe the address range that this mapping
    /// occupies.
    ///
    /// AVMA = "actual virtual memory address"
    ///
    /// `relative_address_at_start` is the "relative address" which corresponds
    /// to `start_avma`, in the library that is mapped in this mapping. This is zero if
    /// `start_avm` is the base address of the library.
    ///
    /// A relative address is a `u32` value which is relative to the library base address.
    /// So you will usually set `relative_address_at_start` to `start_avma - base_avma`.
    ///
    /// For ELF binaries, the base address is the AVMA of the first segment, i.e. the
    /// start_avma of the mapping created by the first ELF `LOAD` command.
    ///
    /// For mach-O binaries, the base address is the vmaddr of the `__TEXT` segment.
    ///
    /// For Windows binaries, the base address is the image load address.
    pub fn add_mapping(
        &mut self,
        start_avma: u64,
        end_avma: u64,
        relative_address_at_start: u32,
        value: T,
    ) where
        T: Clone,
    {
        // Collect the start keys of every existing mapping that intersects
        // `[start_avma, end_avma)`. A mapping `[m_start, m_end)` intersects iff
        // `m_start < end_avma && m_end > start_avma`. Every such mapping starts
        // within the range, except possibly one that starts before `start_avma`
        // and reaches into it — `lookup_impl(start_avma)` finds that one.
        let mut intersecting: Vec<u64> = self
            .map
            .range(start_avma..end_avma)
            .map(|(start_avma, _)| *start_avma)
            .collect();
        if let Some(straddler) = self.lookup_impl(start_avma) {
            if straddler.start_avma < start_avma {
                intersecting.push(straddler.start_avma);
            }
        }

        // Replace each intersecting mapping with its non-overlapping remainder.
        for key in intersecting {
            let Mapping {
                start_avma: m_start,
                end_avma: m_end,
                relative_address_at_start: m_rel,
                value: m_value,
            } = self.map.remove(&key).unwrap();

            // The head `[m_start, start_avma)` keeps the original relative
            // address; the tail `[end_avma, m_end)` starts `end_avma - m_start`
            // bytes further into the library, so its relative address shifts by
            // that amount.
            let head = Mapping {
                start_avma: m_start,
                end_avma: start_avma,
                relative_address_at_start: m_rel,
                value: m_value.clone(),
            };
            let tail = Mapping {
                start_avma: end_avma,
                end_avma: m_end,
                relative_address_at_start: m_rel.wrapping_add((end_avma - m_start) as u32),
                value: m_value,
            };
            match (m_start < start_avma, m_end > end_avma) {
                (true, true) => {
                    self.map.insert(head.start_avma, head);
                    self.map.insert(tail.start_avma, tail);
                }
                (true, false) => {
                    self.map.insert(head.start_avma, head);
                }
                (false, true) => {
                    self.map.insert(tail.start_avma, tail);
                }
                (false, false) => {} // fully covered: drop it
            }
        }

        self.map.insert(
            start_avma,
            Mapping {
                start_avma,
                end_avma,
                relative_address_at_start,
                value,
            },
        );
    }

    /// Remove a mapping which starts at the given address. If found, this returns
    /// the `relative_address_at_start` and the associated value of the mapping.
    pub fn remove_mapping(&mut self, start_avma: u64) -> Option<(u32, T)> {
        self.map
            .remove(&start_avma)
            .map(|m| (m.relative_address_at_start, m.value))
    }

    /// Clear all mappings.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Look up the mapping which covers the given address and return
    /// the stored value.
    pub fn lookup(&self, avma: u64) -> Option<&T> {
        self.lookup_impl(avma).map(|m| &m.value)
    }

    /// Look up the mapping which covers the given address and return
    /// its `Mapping<T>``.
    fn lookup_impl(&self, avma: u64) -> Option<&Mapping<T>> {
        let (_start_avma, last_mapping_starting_at_or_before_avma) =
            self.map.range(..=avma).next_back()?;
        if avma < last_mapping_starting_at_or_before_avma.end_avma {
            Some(last_mapping_starting_at_or_before_avma)
        } else {
            None
        }
    }

    /// Converts an absolute address (AVMA, actual virtual memory address) into
    /// a relative address and the mapping's associated value.
    pub fn convert_address(&self, avma: u64) -> Option<(u32, &T)> {
        let mapping = self.lookup_impl(avma)?;
        let offset_from_mapping_start = (avma - mapping.start_avma) as u32;
        let relative_address = mapping.relative_address_at_start + offset_from_mapping_start;
        Some((relative_address, &mapping.value))
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq)]
struct Mapping<T> {
    start_avma: u64,
    end_avma: u64,
    relative_address_at_start: u32,
    value: T,
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn lookup_covers_ranges_and_respects_gaps() {
        let mut m = LibMappings::new();
        m.add_mapping(0x1000, 0x2000, 0, "a");
        m.add_mapping(0x3000, 0x4000, 0, "b");
        assert_eq!(m.lookup(0x0fff), None);
        assert_eq!(m.lookup(0x1500), Some(&"a"));
        assert_eq!(m.lookup(0x2000), None); // end is exclusive
        assert_eq!(m.lookup(0x2500), None); // gap between mappings
        assert_eq!(m.lookup(0x3500), Some(&"b"));
    }

    #[test]
    fn non_overlapping_neighbours_are_left_untouched() {
        let mut m = LibMappings::new();
        m.add_mapping(0x1000, 0x2000, 0, "a");
        m.add_mapping(0x3000, 0x4000, 0, "b");
        // Fill the gap exactly; neither neighbour should be disturbed.
        m.add_mapping(0x2000, 0x3000, 0, "c");
        assert_eq!(m.lookup(0x1800), Some(&"a"));
        assert_eq!(m.lookup(0x2800), Some(&"c"));
        assert_eq!(m.lookup(0x3800), Some(&"b"));
    }

    #[test]
    fn exact_overlap_replaces_the_value() {
        let mut m = LibMappings::new();
        m.add_mapping(0x1000, 0x2000, 0, "old");
        m.add_mapping(0x1000, 0x2000, 0, "new");
        assert_eq!(m.lookup(0x1500), Some(&"new"));
    }

    #[test]
    fn new_mapping_clips_the_tail_of_an_existing_one() {
        let mut m = LibMappings::new();
        m.add_mapping(0x1000, 0x3000, 0, "old");
        m.add_mapping(0x2000, 0x4000, 0, "new"); // overlaps old's tail
        assert_eq!(m.lookup(0x1500), Some(&"old")); // old's head survives
        assert_eq!(m.lookup(0x2000), Some(&"new"));
        assert_eq!(m.lookup(0x2fff), Some(&"new"));
        assert_eq!(m.lookup(0x3500), Some(&"new"));
    }

    #[test]
    fn new_mapping_clips_the_head_of_an_existing_one_and_fixes_relative_address() {
        let mut m = LibMappings::new();
        // "lib" is based at 0x2000 (relative address 0 there).
        m.add_mapping(0x2000, 0x4000, 0, "lib");
        m.add_mapping(0x1000, 0x3000, 0, "new"); // overlaps lib's head
        assert_eq!(m.lookup(0x2500), Some(&"new"));
        assert_eq!(m.lookup(0x3500), Some(&"lib")); // lib's tail survives
                                                    // The surviving tail starts at 0x3000, which is 0x1000 into "lib", so an
                                                    // address there must still resolve to the correct relative address.
        assert_eq!(m.convert_address(0x3800), Some((0x1800, &"lib")));
    }

    #[test]
    fn mapping_fully_inside_the_new_one_is_dropped() {
        let mut m = LibMappings::new();
        m.add_mapping(0x2000, 0x3000, 0, "inner");
        m.add_mapping(0x1000, 0x4000, 0, "outer");
        assert_eq!(m.lookup(0x1500), Some(&"outer"));
        assert_eq!(m.lookup(0x2500), Some(&"outer"));
    }

    /// The motivating real-world case: `ld.so` is mapped with an over-long
    /// executable mapping that spans the gap where the kernel later drops
    /// `[vdso]`/`[vvar]`. Adding `[vdso]` inside it must not evict the whole
    /// interpreter mapping; the executable code before the gap, and the
    /// remainder after it, must stay resolvable with correct relative addresses.
    #[test]
    fn vdso_inside_ld_so_mapping_keeps_the_interpreter_resolvable() {
        let mut m = LibMappings::new();
        let ld_so_base = 0x7fff_0000_0000u64;
        // ld.so executable mapping, base-relative (relative address 0 at start).
        m.add_mapping(ld_so_base, ld_so_base + 0x40000, 0, "ld.so");
        // [vdso] lands in the inter-segment gap, fully inside the ld.so mapping.
        let vdso_start = ld_so_base + 0x2b000;
        m.add_mapping(vdso_start, vdso_start + 0x2000, 0, "[vdso]");

        // Executable code before the gap still resolves to ld.so...
        assert_eq!(
            m.convert_address(ld_so_base + 0x2000),
            Some((0x2000, &"ld.so"))
        );
        // ...the gap itself now belongs to [vdso]...
        assert_eq!(m.lookup(vdso_start + 0x1000), Some(&"[vdso]"));
        // ...and the remainder after [vdso] is still ld.so, with the relative
        // address preserved across the split (this is what regressed before).
        assert_eq!(
            m.convert_address(ld_so_base + 0x30000),
            Some((0x30000, &"ld.so"))
        );
    }

    #[test]
    fn one_new_mapping_carves_several_existing_ones() {
        let mut m = LibMappings::new();
        m.add_mapping(0x1000, 0x2000, 0, "a"); // clipped on its tail
        m.add_mapping(0x2000, 0x3000, 0, "b"); // fully covered
        m.add_mapping(0x3000, 0x5000, 0, "c"); // clipped on its head
        m.add_mapping(0x1800, 0x4000, 0, "new");
        assert_eq!(m.lookup(0x1400), Some(&"a")); // a's head [0x1000,0x1800)
        assert_eq!(m.lookup(0x1800), Some(&"new"));
        assert_eq!(m.lookup(0x2800), Some(&"new")); // b is gone
        assert_eq!(m.lookup(0x3fff), Some(&"new"));
        assert_eq!(m.lookup(0x4000), Some(&"c")); // c's tail [0x4000,0x5000)
        assert_eq!(m.lookup(0x4800), Some(&"c"));
    }
}
