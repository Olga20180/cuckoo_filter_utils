import unittest

from cuckoo_filter_utils import CuckooFilter, FilterFull


class TestConstruction(unittest.TestCase):
    def test_defaults(self):
        cf = CuckooFilter()
        self.assertEqual(cf.size, 0)
        self.assertEqual(len(cf), 0)

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            CuckooFilter(capacity=0)

    def test_invalid_fingerprint_bits(self):
        with self.assertRaises(ValueError):
            CuckooFilter(fingerprint_bits=2)
        with self.assertRaises(ValueError):
            CuckooFilter(fingerprint_bits=20)

    def test_invalid_max_kicks(self):
        with self.assertRaises(ValueError):
            CuckooFilter(max_kicks=0)


class TestInsertLookup(unittest.TestCase):
    def test_inserted_item_is_present(self):
        cf = CuckooFilter(capacity=64)
        cf.insert(b"hello")
        self.assertIn(b"hello", cf)
        self.assertEqual(cf.size, 1)

    def test_absent_item_may_be_false_positive_but_size_stable(self):
        # With a tiny filter and few items, a false positive is plausible but
        # not guaranteed; we assert only the invariants that hold regardless:
        # size reflects actual inserts, and present items are always found.
        cf = CuckooFilter(capacity=16, fingerprint_bits=4)
        for i in range(4):
            cf.insert(b"item-%d" % i)
        for i in range(4):
            self.assertIn(b"item-%d" % i, cf)
        self.assertEqual(cf.size, 4)

    def test_non_bytes_rejected(self):
        cf = CuckooFilter(capacity=16)
        with self.assertRaises(TypeError):
            cf.insert("not-bytes")
        with self.assertRaises(TypeError):
            "x" in cf  # noqa: B015
        with self.assertRaises(TypeError):
            cf.delete(123)

    def test_insert_empty_bytes(self):
        cf = CuckooFilter(capacity=32)
        cf.insert(b"")
        self.assertIn(b"", cf)
        self.assertEqual(cf.size, 1)


class TestDelete(unittest.TestCase):
    def test_delete_present_item(self):
        cf = CuckooFilter(capacity=64)
        cf.insert(b"alpha")
        self.assertTrue(cf.delete(b"alpha"))
        self.assertEqual(cf.size, 0)
        self.assertNotIn(b"alpha", cf)

    def test_delete_absent_returns_false(self):
        cf = CuckooFilter(capacity=64)
        self.assertFalse(cf.delete(b"never-inserted"))

    def test_delete_one_of_duplicates(self):
        cf = CuckooFilter(capacity=64)
        cf.insert(b"dup")
        cf.insert(b"dup")
        self.assertEqual(cf.size, 2)
        self.assertTrue(cf.delete(b"dup"))
        self.assertEqual(cf.size, 1)
        self.assertIn(b"dup", cf)
        self.assertTrue(cf.delete(b"dup"))
        self.assertEqual(cf.size, 0)
        self.assertFalse(cf.delete(b"dup"))

    def test_delete_then_reinsert(self):
        cf = CuckooFilter(capacity=64)
        cf.insert(b"x")
        cf.delete(b"x")
        cf.insert(b"x")
        self.assertIn(b"x", cf)
        self.assertEqual(cf.size, 1)


class TestCapacity(unittest.TestCase):
    def test_fill_and_overflow(self):
        # Tiny filter to force a FilterFull quickly with low max_kicks.
        cf = CuckooFilter(capacity=4, fingerprint_bits=8, max_kicks=8)
        inserted = 0
        n = 2000
        try:
            for i in range(n):
                cf.insert(b"k-%d" % i)
                inserted += 1
        except FilterFull:
            pass
        # Must have inserted at least the raw slot count before failing.
        self.assertGreaterEqual(inserted, 4)
        self.assertLess(inserted, n)
        # Every item we did insert must be lookup-true.
        for i in range(inserted):
            self.assertIn(b"k-%d" % i, cf)

    def test_load_factor_in_unit_interval(self):
        cf = CuckooFilter(capacity=64)
        for i in range(10):
            cf.insert(b"l-%d" % i)
        lf = cf.load_factor
        self.assertGreater(lf, 0.0)
        self.assertLessEqual(lf, 1.0)


class TestRepr(unittest.TestCase):
    def test_repr_mentions_capacity(self):
        cf = CuckooFilter(capacity=128, fingerprint_bits=8)
        r = repr(cf)
        self.assertIn("CuckooFilter", r)
        self.assertIn("128", r)


if __name__ == "__main__":
    unittest.main()
