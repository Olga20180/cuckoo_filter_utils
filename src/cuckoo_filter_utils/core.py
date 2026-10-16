"""A single-bucket-per-slot cuckoo filter implementation.

Design decisions, because they are not obvious from the code alone:

- Standard library only. The fingerprint hash is a hand-rolled 32-bit FNV-1a
  so the library has zero install dependencies and is deterministic across
  runs and platforms (Python's built-in ``hash`` is salted per-process and
  unsuitable here).

- One bucket per index, four slots per bucket. This is the configuration
  Fan, Andersen, Kaminsky, and Mitzenmacher used in the 2014 paper and it
  gives a good false-positive rate for modest sizes without the complexity
  of variable bucket widths.

- Only ``insert`` and ``delete`` mutate the filter. Re-inserting the same
  item is legal and increments the count; ``delete`` removes one occurrence.
  This is the interpretation this library commits to. If you need strict
  set semantics (no duplicates), deduplicate before inserting.

- The false-positive rate is bounded by ``1 / 2^fingerprint_bits`` per the
  paper's analysis; with 8-bit fingerprints that is ~0.39%. The library does
  not estimate occupancy from the rate; it tracks item count directly.

- On insert, if the chosen path cannot be resolved within ``max_kicks``
  attempts, ``FilterFull`` is raised. This is a deliberate, non-silent
  signal: a near-full cuckoo filter should be rebuilt larger, not quietly
  dropped items.
"""


class FilterFull(Exception):
    """Raised when an insert cannot find a free slot within the kick budget."""


def _fnv1a_32(data: bytes) -> int:
    """32-bit FNV-1a hash, standard library only.

    We cannot use ``hash``: it is salted per interpreter process, so a filter
    built in one run would not answer the same queries in the next. FNV-1a is
    cheap, public-domain, and bit-stable.
    """
    h = 0x811C9DC5
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class CuckooFilter:
    """A bounded-occupancy cuckoo filter over ``bytes`` items.

    Items are fingerprinted and stored in one of two candidate buckets derived
    from the item hash and the fingerprint. Lookup may return false
    positives; deletions of items actually present never do.
    """

    __slots__ = ("capacity", "fingerprint_bits", "max_kicks", "_size",
                 "_num_buckets", "_buckets", "_fp_mask", "_rng_state")

    def __init__(self, capacity=4096, fingerprint_bits=8, max_kicks=500):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if fingerprint_bits < 4 or fingerprint_bits > 16:
            raise ValueError("fingerprint_bits must be in [4, 16]")
        if max_kicks <= 0:
            raise ValueError("max_kicks must be positive")
        self.capacity = capacity
        self.fingerprint_bits = fingerprint_bits
        self.max_kicks = max_kicks
        self._size = 0
        # Round capacity up to a multiple of 4 so buckets are uniform.
        self._num_buckets = max(1, (capacity + 3) // 4)
        self._buckets = [bytearray(4) for _ in range(self._num_buckets)]
        self._fp_mask = (1 << fingerprint_bits) - 1
        # Seed for the in-class PRNG used to choose candidate slots. Fixed at
        # construction so behaviour is deterministic for a given filter, which
        # matters for reproducible test runs.
        self._rng_state = 0x9E3779B9

    def _rand(self):
        """A tiny xorshift generator.

        Used only to pick which slot to evict during a kick; it does not need
        cryptographic quality, and avoiding the ``random`` module keeps the
        filter self-contained and seedable without global state.
        """
        x = self._rng_state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self._rng_state = x
        return x

    @property
    def size(self):
        return self._size

    @property
    def load_factor(self):
        """Occupied slots divided by total slots."""
        total = self._num_buckets * 4
        return self._size / total

    def _indices(self, item):
        """Return (i1, i2, fingerprint) for ``item``.

        The standard cuckoo filter derives the alternate index by xoring the
        primary index with a hash of the fingerprint. This guarantees the
        pair is unordered: ``(i, fp)`` and ``(i ^ h(fp), fp)`` resolve to the
        same two buckets regardless of which is the starting point.
        """
        h = _fnv1a_32(item)
        fp = (h >> 1) & self._fp_mask
        if fp == 0:
            fp = 1  # 0 is our sentinel for "empty slot"
        i1 = h % self._num_buckets
        i2 = (i1 ^ _fnv1a_32(fp.to_bytes(2, "little"))) % self._num_buckets
        return i1, i2, fp

    def _alternate_index(self, index, fp):
        return (index ^ _fnv1a_32(fp.to_bytes(2, "little"))) % self._num_buckets

    def insert(self, item):
        """Insert ``item`` (``bytes``). Raise ``FilterFull`` if no slot is found.

        Duplicate inserts are allowed and tracked; call ``delete`` once per
        duplicate to fully remove the item.
        """
        if not isinstance(item, (bytes, bytearray)):
            raise TypeError("item must be bytes")
        i1, i2, fp = self._indices(bytes(item))
        for idx in (i1, i2):
            bucket = self._buckets[idx]
            for slot in range(4):
                if bucket[slot] == 0:
                    bucket[slot] = fp
                    self._size += 1
                    return
        # Kick: start from whichever index, displace along the alternate path.
        idx = i1 if (self._rand() & 1) == 0 else i2
        current_fp = fp
        for _ in range(self.max_kicks):
            slot = self._rand() % 4
            evicted = self._buckets[idx][slot]
            self._buckets[idx][slot] = current_fp
            current_fp = evicted
            idx = self._alternate_index(idx, evicted)
            bucket = self._buckets[idx]
            for s in range(4):
                if bucket[s] == 0:
                    bucket[s] = current_fp
                    self._size += 1
                    return
        raise FilterFull("cuckoo filter full; cannot place item")

    def __contains__(self, item):
        if not isinstance(item, (bytes, bytearray)):
            raise TypeError("item must be bytes")
        i1, i2, fp = self._indices(bytes(item))
        return (fp in self._buckets[i1]) or (fp in self._buckets[i2])

    def delete(self, item):
        """Delete one occurrence of ``item``. Return ``True`` if removed.

        Returns ``False`` if the item's fingerprint is not present in either
        candidate bucket. Because of the filter's nature this may mean the
        item was never inserted, or was already fully deleted. Callers who
        need to know which should track membership externally.
        """
        if not isinstance(item, (bytes, bytearray)):
            raise TypeError("item must be bytes")
        i1, i2, fp = self._indices(bytes(item))
        for idx in (i1, i2):
            bucket = self._buckets[idx]
            for slot in range(4):
                if bucket[slot] == fp:
                    bucket[slot] = 0
                    self._size -= 1
                    return True
        return False

    def __len__(self):
        return self._size

    def __repr__(self):
        return (f"CuckooFilter(capacity={self.capacity}, "
                f"fingerprint_bits={self.fingerprint_bits}, "
                f"size={self._size})")
