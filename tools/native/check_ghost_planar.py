#!/usr/bin/env python3
"""WP7 defined selected stock B versus same-source observed selected C.

The old planar ghost route A has initialized IDs, unlike the old 3D ghost
route.  Where supplied, A is a separately built baseline for public native
geometry comparison.  Its output is never an occurrence-provenance oracle.
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

import numpy as np

from qualification.native_import import load_production_module


def load(path, label):
    if label == 'production':
        return load_production_module(path, 'pyvoro2._core2d')
    spec = importlib.util.spec_from_file_location(f'{label}._core2d', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bits(value):
    if isinstance(value, float):
        return struct.pack('>d', value).hex()
    if isinstance(value, dict):
        return {key: bits(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [bits(item) for item in value]
    return value


def corpus():
    for mask, power in itertools.product(itertools.product((False, True), repeat=2),
                                         (False, True)):
        for label, points, queries in (
            ('ghost-only', [], [[.375, .625]]),
            ('sparse-batch', [[.125, .25], [.75, .875]], [[.375, .375], [.5, .5]]),
            ('repeated', [[.125, .25]], [[.75, .75], [.75, .75]]),
        ):
            yield dict(name=f'{mask}-{power}-{label}', mask=mask,
                       points=points, queries=queries,
                       radii=([.1] * len(points) if power else None),
                       ghost_radii=([.2] * len(queries) if power else None),
                       bounds=((0., 1.), (0., 1.)), blocks=(1, 1),
                       init_mem=1)
    yield dict(name='power-hidden', mask=(True, True), points=[[.25, .5]],
               queries=[[.75, .5]], radii=[4.], ghost_radii=[0.],
               bounds=((0., 1.), (0., 1.)), blocks=(1, 1), init_mem=1)
    yield dict(name='slot-growth', mask=(False, False),
               points=[[math.cos(2 * math.pi * k / 80),
                        math.sin(2 * math.pi * k / 80)] for k in range(80)],
               queries=[[0., 0.]], radii=None, ghost_radii=None,
               bounds=((-2., 2.), (-2., 2.)), blocks=(1, 1), init_mem=1)
    yield dict(name='ghost-step-remap', mask=(True, False), points=[],
               queries=[[math.nextafter(.1, 0.), .5]], radii=None,
               ghost_radii=None, bounds=((0., .1), (0., 1.)),
               blocks=(5, 1), init_mem=1, changed_site=True)
    for power in (False, True):
        yield dict(name=f'ghost-omitted-{power}', mask=(False, True), points=[],
                   queries=[[math.nextafter(1., 0.), .5]],
                   radii=[] if power else None,
                   ghost_radii=[0.] if power else None,
                   bounds=((-1., 1.), (0., 1.)), blocks=(1, 1),
                   init_mem=1, omitted=True)
    yield dict(name='complete-batch-resource', mask=(True, True), points=[],
               queries=[[.5, .5]] * 120_000, radii=None, ghost_radii=None,
               bounds=((0., 1.), (0., 1.)), blocks=(1, 1),
               init_mem=1, batch_resource=True)


def arguments(spec):
    p = np.asarray(spec['points'], dtype=float).reshape(-1, 2)
    q = np.asarray(spec['queries'], dtype=float).reshape(-1, 2)
    args = [p, np.arange(len(p), dtype=np.int32)]
    if spec['radii'] is not None:
        args.append(np.asarray(spec['radii'], dtype=float))
    args.extend([spec['bounds'], spec['blocks'], spec['mask'],
                 spec['init_mem'], (True, True, True), q])
    if spec['radii'] is not None:
        args.append(np.asarray(spec['ghost_radii'], dtype=float))
    return args


def geometry(packet):
    return [{key: source[key] for key in ('id', 'present', 'local2', 'next')}
            for source in packet['sources']]


def run(module, prefix, spec, suffix=''):
    mode = 'power' if spec['radii'] is not None else 'standard'
    return getattr(module, f'{prefix}{mode}{suffix}')(*arguments(spec))


def main():
    if sys.flags.optimize or not __debug__:
        raise RuntimeError('qualification requires enabled Python assertions')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--production-module', type=Path,
                        help='same-source release build; require qualified profile')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    module = load(args.module, 'candidate')
    production = (load(args.production_module, 'production')
                  if args.production_module else None)
    baseline = load(args.baseline, 'baseline') if args.baseline else None
    profile = module._planar_witness_profile()
    source_sha = hashlib.sha256(args.source_manifest.read_bytes()).hexdigest()
    assert profile['source_sha256'] == source_sha
    if production:
        assert production._planar_witness_profile()['source_sha256'] == source_sha
        from pyvoro2._internal.native_qualification import require_native
        require_native(production, 'wp7-planar')
    results = []
    for fixture in corpus():
        if fixture.get('batch_resource'):
            for prefix in ('_planar_ghost_stock_', '_planar_ghost_candidate_'):
                try:
                    run(module, prefix, fixture)
                except RuntimeError as error:
                    assert 'planar_certification:resource:batch_observer' in str(error)
                else:
                    raise AssertionError('unbounded complete ghost batch accepted')
            if production is not None:
                try:
                    run(production, '_ghost_box_', fixture, '_witness')
                except RuntimeError as error:
                    assert 'planar_certification:resource:batch_observer' in str(error)
                else:
                    raise AssertionError('unbounded production ghost batch accepted')
            results.append({'name': fixture['name'], 'batch_resource_refused': True})
            continue
        if fixture.get('omitted'):
            for prefix in ('_planar_ghost_stock_', '_planar_ghost_candidate_'):
                try:
                    run(module, prefix, fixture)
                except RuntimeError as error:
                    assert 'planar_certification:insertion:omitted' in str(error)
                else:
                    raise AssertionError(f'{fixture["name"]}: omitted ghost accepted')
            if production is not None:
                try:
                    run(production, '_ghost_box_', fixture, '_witness')
                except RuntimeError as error:
                    assert 'planar_certification:insertion:omitted' in str(error)
                else:
                    raise AssertionError(
                        f'{fixture["name"]}: production omitted ghost accepted')
            results.append({'name': fixture['name'], 'omission_refused': True})
            continue
        stock_cells, stock_packets = run(module, '_planar_ghost_stock_', fixture)
        observed_cells, observed_packets = run(
            module, '_planar_ghost_candidate_', fixture)
        assert bits(stock_cells) == bits(observed_cells), fixture['name']
        assert len(stock_packets) == len(observed_packets) == len(fixture['queries'])
        for index, (stock, observed) in enumerate(zip(stock_packets, observed_packets)):
            assert bits(geometry(stock)) == bits(geometry(observed)), fixture['name']
            assert bits(stock['inserted']) == bits(observed['inserted']), (
                fixture['name'])
            assert observed['ghost_internal_id'] == len(fixture['points'])
            assert observed['query_index'] == index
            source, = observed['sources']
            for origin in source['origins']:
                assert origin['source'] == source['id']
                if origin['kind'] == 'particle':
                    assert 0 <= origin['owner'] <= observed['ghost_internal_id']
                    assert all(-1 <= value <= 1 for value in origin['sigma'])
                    assert all(fixture['mask'][axis] or value == 0
                               for axis, value in enumerate(origin['sigma']))
                else:
                    assert origin['kind'] == 'initialization'
                    assert origin['side'] in {-1, -2, -3, -4}
        result = {'name': fixture['name'], 'stock_observed_bits': True,
                  'sources': len(observed_packets),
                  'occurrences': sum(len(row['sources'][0]['origins'])
                                     for row in observed_packets)}
        if baseline is not None and not fixture.get('changed_site'):
            old = run(baseline, 'ghost_box_', fixture)
            assert bits(old) == bits(stock_cells), fixture['name']
            result['baseline_stock_serialization_bits'] = True
        if production is not None:
            final_cells, final_packets = run(
                production, '_ghost_box_', fixture, '_witness')
            assert bits(final_cells) == bits(observed_cells), fixture['name']
            for final, candidate in zip(final_packets, observed_packets):
                final = dict(final)
                candidate = dict(candidate)
                final.pop('profile')
                candidate.pop('profile')
                assert bits(final) == bits(candidate), fixture['name']
            result['production_observed_bits'] = True
        results.append(result)
    report = {'profile': profile, 'source_manifest_sha256': source_sha,
              'modules': {'candidate': hashlib.sha256(
                  args.module.read_bytes()).hexdigest()},
              'cases': results}
    if baseline:
        report['modules']['baseline'] = hashlib.sha256(
            args.baseline.read_bytes()).hexdigest()
    if production:
        report['modules']['production'] = hashlib.sha256(
            args.production_module.read_bytes()).hexdigest()
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(json.dumps({'cases': len(results), 'source_sha256': source_sha,
                      'stock_observed': True}))


if __name__ == '__main__':
    main()
