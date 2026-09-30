#!/usr/bin/env python3
"""WP8 stock/observed same-call parity on a fixed, bounded native corpus.

This tool does not assert an exact ownership theorem. It compares unchanged
native selections and positions to the pre-WP8 module, including copied images,
query remaps, insertion branches, power radii and storage growth. Independent
exact public image expectations live in the WP8 tests.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

from qualification.native_import import load_production_module


def load(path, name):
    if name in ('pyvoro2._core', 'pyvoro2._core2d'):
        return load_production_module(path, name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('module2', 'module3', 'baseline2', 'baseline3', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    results = []
    for dim in (2, 3):
        suffix = '_core2d' if dim == 2 else '_core'
        current = load(getattr(args, f'module{dim}'), f'pyvoro2.{suffix}')
        stock = load(getattr(args, f'baseline{dim}'), f'stock{dim}.{suffix}')
        rng = np.random.default_rng(7900 + dim)
        for power, mask, blocks in itertools.product(
                (False, True), itertools.product((False, True), repeat=dim),
                ((1,) * dim, (3,) * dim)):
            points = rng.integers(32, 224, size=(19, dim)).astype(float) / 256
            queries = rng.integers(-700, 950, size=(25, dim)).astype(float) / 256
            ids = np.arange(len(points), dtype=np.int32)
            radii = np.arange(len(points), dtype=float) / 128
            call_args = [points, ids] + ([radii] if power else [])
            call_args += [((0., 1.),) * dim, blocks, mask, 1, queries]
            name = 'locate_box_' + ('power' if power else 'standard')
            old = getattr(stock, name)(*call_args)
            new = getattr(current, name)(*call_args, return_source=True)
            assert all(a.tobytes() == b.tobytes() for a, b in zip(old, new[:3]))
            assert new[3]['stored'].shape == points.shape
            results.append({'dim': dim, 'route': name, 'mask': mask,
                            'blocks': blocks, 'queries': len(queries),
                            'bitwise_stock_equal': True})
        if dim == 3:
            for power, params, blocks in itertools.product(
                    (False, True), ((1., .375, 1., -.25, .125, 1.),
                                    (1., 2., 1., -3., 2., 1.)),
                    ((1, 1, 1), (3, 3, 3))):
                points = rng.integers(32, 224, size=(19, 3)).astype(float) / 256
                queries = rng.integers(-700, 950, size=(25, 3)).astype(float) / 256
                call_args = [points, np.arange(len(points), dtype=np.int32)]
                if power:
                    call_args.append(np.arange(len(points), dtype=float) / 128)
                call_args += [params, blocks, 1, queries]
                name = 'locate_periodic_' + ('power' if power else 'standard')
                old = getattr(stock, name)(*call_args)
                new = getattr(current, name)(*call_args, return_source=True)
                assert all(a.tobytes() == b.tobytes() for a, b in zip(old, new[:3]))
                results.append({'dim': dim, 'route': name, 'params': params,
                                'blocks': blocks, 'queries': len(queries),
                                'bitwise_stock_equal': True})
    modules = {
        name: {'path': str(getattr(args, name)), 'sha256': digest(getattr(args, name))}
        for name in ('module2', 'module3', 'baseline2', 'baseline3')
    }
    report = {'modules': modules, 'cases': results,
              'total_queries': sum(r['queries'] for r in results)}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(results)} cases, {report["total_queries"]} queries: bitwise parity')


if __name__ == '__main__':
    main()
