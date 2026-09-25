#!/usr/bin/env python3
"""Compare stock A, FP-qualified B, and optimized compact WP6 observer C.

The optional archive comparison reads the independently reviewed prototype's
raw inputs and final origin associations, never the Python production certifier.
B's stock packet probe is compiled only with PYVORO2_PLANAR_QUALIFICATION=ON.
A candidate cohort can be measured explicitly; successful measurements do not
edit the support predicate or qualify that cohort automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import time

import numpy as np


def load(path, label):
    spec = importlib.util.spec_from_file_location(f"{label}._core2d", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact(value):
    """Lossless binary64 comparison, including the sign of zero."""
    if isinstance(value, float):
        return struct.pack(">d", value).hex()
    if isinstance(value, dict):
        return {key: exact(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact(val) for val in value]
    return value


def decode(value):
    if isinstance(value, str) and (value.startswith("0x") or
                                   value.startswith("-0x")):
        return float.fromhex(value)
    if isinstance(value, dict):
        return {key: decode(val) for key, val in value.items()}
    if isinstance(value, list):
        return [decode(val) for val in value]
    return value


def corpus():
    for mask in itertools.product((False, True), repeat=2):
        for power in (False, True):
            for name, points in (
                ("empty", []), ("one", [[.375, .625]]),
                ("sparse", [[.125, .25], [.75, .25], [.375, .875]]),
                ("diamond", [[.25, .25], [.75, .75]]),
            ):
                yield {"name": f"{mask}-{power}-{name}", "points": points,
                       "radii": [0.] * len(points) if power else None,
                       "bounds": [[0., 1.], [0., 1.]], "blocks": [1, 1],
                       "periodic": mask}
    for blocks in ([1, 1], [5, 5], [24, 24]):
        for radius in (0., 2.**27, 2.**27 + 1):
            yield {"name": f"radius-{radius}-grid-{blocks[0]}",
                   "points": [[.125, .25], [.75, .25], [.375, .875]],
                   "radii": [radius] * 3, "bounds": [[0., 1.]] * 2,
                   "blocks": blocks, "periodic": [True, True]}
    yield {"name": "capacity-growth", "points": [[0., 0.]] + [
        [math.cos(2 * math.pi * i / 270), math.sin(2 * math.pi * i / 270)]
        for i in range(270)], "radii": None, "bounds": [[-2., 2.]] * 2,
        "blocks": [1, 1], "periodic": [False, False]}
    yield {"name": "hidden", "points": [[.25, .5], [.75, .5]],
           "radii": [0., 2.], "bounds": [[0., 1.]] * 2,
           "blocks": [1, 1], "periodic": [True, True]}
    yield {"name": "coincident", "points": [[.25, .5], [.75, .5]],
           "radii": [.75, .25], "bounds": [[0., 1.]] * 2,
           "blocks": [1, 1], "periodic": [False, True]}
    yield {"name": "private-eps0-insertion-remap",
           "points": [[math.nextafter(.1, 0.), .5]], "radii": None,
           "bounds": [[0., .1], [0., 1.]], "blocks": [5, 1],
           "periodic": [True, False]}
    yield {"name": "signed-zero", "points": [[-0., .5]], "radii": None,
           "bounds": [[-0., 1.], [0., 1.]], "blocks": [1, 1],
           "periodic": [True, True]}
    for power in (False, True):
        yield {"name": f"intended-insertion-refusal-{power}",
               "points": [[math.nextafter(1., 0.), .5]],
               "radii": [0.] if power else None,
               "bounds": [[-1., 1.], [0., 1.]], "blocks": [1, 1],
               "periodic": [False, True], "omitted": True}


def archived(directory):
    # The fixture index names exactly the ordinary computations.  The separate
    # manual-clipping packet is not an ordinary container run.
    index = json.loads((directory / "fixtures" / "inputs.json").read_text())
    for entry in index:
        path = directory / "native-packets" / (entry["name"] + ".json")
        raw = json.loads(path.read_text())
        data = decode(raw)
        prep, spec = data["preparation"], data["fixture"]
        yield {"name": spec["name"], "points": prep["prepared"],
               "radii": (None if spec["mode"] == "standard"
                         else prep["backend_radii"]),
               "bounds": spec["bounds"], "blocks": spec["blocks"],
               "periodic": spec["periodic"],
               "omitted": spec["name"].startswith("ordinary_insertion_omission"),
               "archive": data}


def arguments(spec):
    points = np.asarray(spec["points"], dtype=float).reshape(-1, 2)
    args = [points, np.arange(len(points), dtype=np.int32)]
    if spec["radii"] is not None:
        args.append(np.asarray(spec["radii"], dtype=float))
    args.extend([spec["bounds"], spec["blocks"], spec["periodic"], 1,
                 (True, True, True)])
    return args


def native_view(packet):
    return [{key: row[key] for key in ("id", "present", "local2", "next")}
            for row in packet["sources"]]


def compare_archive(packet, spec):
    archive = spec["archive"]
    old = archive["packets"]["C"]
    traces = {row["id"]: row for row in old["traces"]}
    originals = {row["id"]: row for row in old["native"]}
    for source in packet["sources"]:
        original = originals[source["id"]]
        assert source["present"] == original["present"]
        if not source["present"]:
            continue
        values = [struct.pack(">d", value).hex()
                  for point in source["local2"] for value in point]
        assert values == original["cell"]["pts_bits"]
        assert source["next"] == original["cell"]["ed"][::2]
        trace = traces[source["id"]]
        origins = {row["token"]: row for row in trace["origins"]}
        for row, token in zip(source["origins"], trace["final"]["ne"]):
            expected = origins[token]
            assert row["source"] == expected["source"]
            assert row["kind"] == expected["kind"]
            if row["kind"] == "initialization":
                assert row["side"] == expected["side_code"]
            else:
                assert row["owner"] == expected["owner"]
                assert list(row["sigma"]) == expected["sigma"]
    actual = {row["id"]: row for row in packet["inserted"]}
    for stored in old["storage"]["rows"]:
        row = actual[stored["id"]]
        assert row["block"] == stored["block"]
        assert row["slot"] == stored["slot"]
        values = list(row["point"])
        if spec["radii"] is not None:
            values.append(row["radius"])
        assert exact(values) == stored["stored_bits"]
    for row in old["insertion_replay"]:
        assert list(actual[row["id"]]["h"]) == row["h_if_inserted"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True,
                        help="final production _core2d shared module")
    parser.add_argument("--qualification-module", type=Path, required=True,
                        help="same-source build with qualification probes ON")
    parser.add_argument("--baseline", type=Path,
                        help="stock original A module; records build-profile effects")
    parser.add_argument("--archive", type=Path,
                        help="reviewed archive evidence directory, with native-packets")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--candidate-cohort", action="store_true",
                        help="measure unqualified cohort; never marks it supported")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    production = load(args.module, "production")
    qualified = load(args.qualification_module, "qualified")
    baseline = load(args.baseline, "baseline") if args.baseline else None
    source_sha = hashlib.sha256(args.source_manifest.read_bytes()).hexdigest()
    profile = production._planar_witness_profile()
    assert profile["source_sha256"] == source_sha
    assert qualified._planar_witness_profile()["source_sha256"] == source_sha
    if not args.candidate_cohort:
        assert profile["qualified"], profile
    fixtures = archived(args.archive) if args.archive else corpus()
    results = []
    started = time.perf_counter()
    for fixture in fixtures:
        mode = "standard" if fixture["radii"] is None else "power"
        stock = getattr(qualified, f"_planar_stock_{mode}")
        observer = (getattr(qualified, f"_planar_candidate_{mode}")
                    if args.candidate_cohort else
                    getattr(production, f"_compute_box_{mode}_witness"))
        call_args = arguments(fixture)
        row = {"name": fixture["name"]}
        if fixture.get("omitted"):
            for fn in (stock, observer):
                try:
                    fn(*call_args)
                except RuntimeError as error:
                    assert "planar_certification:insertion:omitted" in str(error)
                else:
                    raise AssertionError("incomplete inserted population accepted")
            row["intentional_insertion_refusal"] = True
            results.append(row)
            continue
        b_cells, b = stock(*call_args)
        c_cells, c = observer(*call_args)
        assert exact(b_cells) == exact(c_cells), fixture["name"]
        assert exact(native_view(b)) == exact(native_view(c)), fixture["name"]
        assert exact(b["inserted"]) == exact(c["inserted"]), fixture["name"]
        row["B_C_native_storage_serialization"] = True
        if baseline:
            a_cells = getattr(baseline, f"compute_box_{mode}")(*call_args)
            assert exact(a_cells) == exact(b_cells), fixture["name"]
            row["A_B_serialization"] = True
        if "archive" in fixture:
            compare_archive(c, fixture)
            row["archived_source_origins_storage"] = True
        row["sources"] = len(c["sources"])
        row["occurrences"] = sum(len(s["origins"]) for s in c["sources"])
        results.append(row)
    report = {"profile": profile, "candidate_only": args.candidate_cohort,
              "elapsed_seconds": time.perf_counter() - started,
              "modules": {
                  "production": hashlib.sha256(args.module.read_bytes()).hexdigest(),
                  "qualified": hashlib.sha256(
                      args.qualification_module.read_bytes()).hexdigest()},
              "source_manifest_sha256": source_sha, "cases": results}
    if args.baseline:
        report["modules"]["baseline"] = hashlib.sha256(
            args.baseline.read_bytes()).hexdigest()
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output)
    print(json.dumps({"cases": len(results), "candidate_only": args.candidate_cohort,
                      "seconds": report["elapsed_seconds"],
                      "source_sha256": source_sha}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
