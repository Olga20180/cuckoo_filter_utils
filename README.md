# cuckoo_filter_utils

A small, dependency-free cuckoo filter for testing set membership with dynamic
deletion, using cuckoo hashing and short fingerprints.

```python
from cuckoo_filter_utils import CuckooFilter, FilterFull

cf = CuckooFilter(capacity=4096, fingerprint_bits=8)
cf.insert(b"user:42")
print(b"user:42" in cf)   # True
cf.delete(b"user:42")
print(b"user:42" in cf)   # False
```

## Why this exists

The need was a bounded, mutable approximate-membership structure that supports
deletion, unlike a classic Bloom filter. A counting Bloom filter also supports
deletion but uses several bits per slot and still cannot reclaim space after
items are removed; a cuckoo filter stores a fingerprint per item and evicts
along a hash-derived alternate path, so deletions are cheap and occupancy stays
honest. The trade-off: the false-positive rate is bounded by the fingerprint
width, and a nearly full filter will eventually refuse inserts.

## Edge you will hit

Inserts can raise ``FilterFull`` once the filter approaches capacity. This is
intentional and non-silent: rebuild with a larger capacity rather than
silently dropping items. Duplicate inserts are counted, so call ``delete``
once per duplicate to fully remove an item.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

