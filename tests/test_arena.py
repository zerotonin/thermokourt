"""Basic tests for ThermoKourt arena extraction."""

import json
import pytest

from thermokourt.extract.arena_extractor import (
    Arena,
    sort_arenas_row_major,
)


class TestArena:
    def test_bbox(self):
        a = Arena(cx=100.0, cy=100.0, r=50.0, idx=0)
        x, y, w, h = a.bbox()
        assert x == 50
        assert y == 50
        assert w == 100
        assert h == 100

    def test_bbox_clamps_negative(self):
        a = Arena(cx=10.0, cy=10.0, r=50.0, idx=0)
        x, y, w, h = a.bbox()
        assert x == 0
        assert y == 0

    def test_to_dict_roundtrip(self):
        a = Arena(cx=123.4, cy=567.8, r=90.1, idx=3)
        d = a.to_dict()
        a2 = Arena.from_dict(d)
        assert a2.cx == a.cx
        assert a2.cy == a.cy
        assert a2.r == a.r
        assert a2.idx == a.idx


class TestSortArenas:
    def test_row_major_order(self):
        # Two rows of 3 arenas, deliberately unsorted
        arenas = [
            Arena(cx=300, cy=100, r=40, idx=0),  # row 0, col 2
            Arena(cx=100, cy=100, r=40, idx=0),  # row 0, col 0
            Arena(cx=200, cy=100, r=40, idx=0),  # row 0, col 1
            Arena(cx=200, cy=250, r=40, idx=0),  # row 1, col 1
            Arena(cx=100, cy=250, r=40, idx=0),  # row 1, col 0
        ]
        sorted_a = sort_arenas_row_major(arenas)
        assert len(sorted_a) == 5
        # Check indices assigned correctly
        assert [a.idx for a in sorted_a] == [0, 1, 2, 3, 4]
        # Check order: row 0 left-to-right, then row 1
        assert sorted_a[0].cx == 100 and sorted_a[0].cy == 100
        assert sorted_a[1].cx == 200 and sorted_a[1].cy == 100
        assert sorted_a[2].cx == 300 and sorted_a[2].cy == 100
        assert sorted_a[3].cx == 100 and sorted_a[3].cy == 250
        assert sorted_a[4].cx == 200 and sorted_a[4].cy == 250

    def test_empty(self):
        assert sort_arenas_row_major([]) == []
