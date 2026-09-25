"""Native anchors derived independently of the production WP6 certifier."""
from __future__ import annotations

import ctypes
import math
import struct
import sys

import numpy as np
import pytest

from pyvoro2 import _core2d


MASKS = [(False, False), (True, False), (False, True), (True, True)]
BOUNDS = ((0.0, 1.0), (0.0, 1.0))


def witness(points, *, radii=None, periodic=(True, True), bounds=BOUNDS,
            blocks=(1, 1), ids=None, opts=(True, True, True), init_mem=1):
    points = np.asarray(points, dtype=float).reshape((-1, 2))
    if ids is None:
        ids = np.arange(len(points), dtype=np.int32)
    name = ("_compute_box_standard_witness" if radii is None
            else "_compute_box_power_witness")
    assert hasattr(_core2d, name), "compact source-associated witness is missing"
    args = [points, np.asarray(ids, dtype=np.int32)]
    if radii is not None:
        args.append(np.asarray(radii, dtype=float))
    return getattr(_core2d, name)(
        *args, bounds, blocks, periodic, init_mem, opts)


def bits(value):
    return struct.pack("d", value)


@pytest.mark.parametrize("periodic", MASKS)
@pytest.mark.parametrize("radii", [None, [0.375]])
def test_one_site_retains_four_initialization_origins(periodic, radii):
    cells, packet = witness([[0.25, 0.5]], periodic=periodic, radii=radii)
    assert len(cells) == len(packet["inserted"]) == 1
    source, = packet["sources"]
    assert source["id"] == 0 and source["present"] is True
    assert {row["side"] for row in source["origins"]} == {-1, -2, -3, -4}
    assert all(row["kind"] == "initialization" for row in source["origins"])
    assert len(source["local2"]) == 4
    for slot, origin in enumerate(source["origins"]):
        assert origin["source"] == source["id"]
        assert origin["slot"] == slot
        assert origin["next"] == source["next"][slot]
        assert cells[0]["edges"][slot]["adjacent_cell"] == origin["side"]
        assert cells[0]["edges"][slot]["vertices"] == [slot, origin["next"]]
    assert tuple(packet["periodic"]) == periodic
    assert tuple(packet["periods"]) == (1.0, 1.0)


@pytest.mark.parametrize("periodic", MASKS)
def test_empty_population_has_complete_empty_packet(periodic):
    cells, packet = witness([], periodic=periodic)
    assert cells == packet["sources"] == packet["inserted"] == []


def test_diamond_four_images_and_collapsed_initialization():
    _, packet = witness([[0.25, 0.25], [0.75, 0.75]])
    source = next(row for row in packet["sources"] if row["id"] == 0)
    other = [row for row in source["origins"] if row["kind"] == "particle"]
    assert {(row["owner"], tuple(row["sigma"])) for row in other} == {
        (1, (-1, -1)), (1, (-1, 0)), (1, (0, -1)), (1, (0, 0)),
    }
    for row in source["origins"]:
        a = source["local2"][row["slot"]]
        b = source["local2"][row["next"]]
        assert (a == b) == (row["kind"] == "initialization")


def test_wall_coincident_noop_retains_initialization():
    _, packet = witness([[0.25, 0.5], [0.75, 0.5]], radii=[0.75, 0.25],
                        periodic=(False, True))
    source = next(row for row in packet["sources"] if row["id"] == 0)
    assert any(row["kind"] == "initialization" and row["side"] == -2
               for row in source["origins"])


@pytest.mark.parametrize("power", [False, True])
@pytest.mark.parametrize("observed", [False, True])
def test_omitted_insertion_refuses_before_hidden_cell_interpretation(
        power, observed):
    points = np.array([[math.nextafter(1.0, 0.0), 0.5]])
    if observed:
        with pytest.raises(RuntimeError,
                           match=r"planar_certification:insertion:omitted"):
            witness(points, radii=[0.0] if power else None,
                    periodic=(False, False), bounds=((-1.0, 1.0), (0.0, 1.0)),
                    opts=(False, False, False))
    else:
        fn = (_core2d.compute_box_power if power
              else _core2d.compute_box_standard)
        args = [points, np.array([0], dtype=np.int32)]
        if power:
            args.append(np.array([0.0]))
        with pytest.raises(RuntimeError,
                           match=r"planar_certification:insertion:omitted"):
            fn(*args, ((-1.0, 1.0), (0.0, 1.0)), (1, 1),
               (False, False), 1, (False, False, False))


def test_actual_insertion_transport_and_signed_zero_storage():
    # Archived private eps=0 stress, not a default compute preparation history.
    _, packet = witness([[math.nextafter(0.1, 0.0), 0.5]],
                        bounds=((0.0, 0.1), (0.0, 1.0)), blocks=(5, 1))
    inserted, = packet["inserted"]
    assert tuple(inserted["h"]) == (1, 0)
    assert bits(inserted["point"][0]) == bits(-2.0**-56)
    assert inserted["block"] == inserted["slot"] == 0
    _, packet = witness([[-0.0, 0.5]])
    assert bits(packet["inserted"][0]["point"][0]) == bits(0.0)


def test_hidden_source_still_has_inserted_record():
    _, packet = witness([[0.25, 0.5], [0.75, 0.5]], radii=[0.0, 4.0])
    assert {row["id"] for row in packet["inserted"]} == {0, 1}
    source = next(row for row in packet["sources"] if row["id"] == 0)
    assert source["present"] is False
    assert source["local2"] == source["next"] == source["origins"] == []


def test_internal_geometry_without_public_geometry():
    cells, packet = witness([[0.25, 0.25], [0.75, 0.75]],
                            opts=(False, False, True))
    assert all("vertices" not in row and "adjacency" not in row
               for row in cells)
    assert all(row["local2"] and row["origins"]
               for row in packet["sources"])


def test_ids_must_be_unambiguous_internal_indices():
    with pytest.raises((ValueError, RuntimeError),
                       match="duplicate internal IDs"):
        witness([[0.25, 0.25], [0.75, 0.75]], ids=[0, 0])


def test_profile_binds_schema_source_and_effective_fp_contract():
    assert hasattr(_core2d, "_planar_witness_profile")
    from pyvoro2._internal.planar.wp6_profile import require_supported_profile
    profile = _core2d._planar_witness_profile()
    require_supported_profile(profile)
    assert profile["qualified"] is True
    assert profile["flt_eval_method"] == 0
    assert profile["int_bits"] == profile["uint_bits"] == 32
    assert profile["double_digits"] == 53
    assert profile["round_to_nearest"] and profile["gradual_underflow"]
    assert profile["fp_contract"] == "off"
    assert profile["fast_math"] is False and profile["lto"] is False
    with pytest.raises(RuntimeError,
                       match=r"planar_certification:profile:source"):
        require_supported_profile(dict(profile, source_sha256="0" * 64))
    with pytest.raises(RuntimeError,
                       match=r"planar_certification:profile:schema"):
        require_supported_profile(dict(profile, schema="invented"))


@pytest.mark.skipif(sys.platform != "linux", reason="uses glibc fenv constants")
def test_changed_runtime_rounding_is_refused_and_restored():
    assert hasattr(_core2d, "_compute_box_standard_witness")
    lib = ctypes.CDLL(None)
    old = lib.fegetround()
    try:
        assert lib.fesetround(0x400) == 0
        with pytest.raises(RuntimeError,
                           match=r"planar_certification:profile:rounding"):
            witness([[0.25, 0.5]])
    finally:
        assert lib.fesetround(old) == 0


def test_observer_storage_budget_refuses_before_native_allocations():
    with pytest.raises(RuntimeError,
                       match="planar_certification:resource:observer_storage"):
        witness([[0.25, 0.5]], blocks=(2200, 2200))
