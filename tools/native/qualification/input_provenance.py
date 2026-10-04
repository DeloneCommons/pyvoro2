"""Independently establish installed header providers for actual dependencies.

Candidate include/search flags never establish a provider. The controlled
compiler and Python installation are trusted toolchain inputs; repository TUs
and headers still require membership in the measured source closure.
"""
from __future__ import annotations

import base64
import importlib.metadata
from pathlib import Path
import re
import subprocess
import sys
import sysconfig

from .effective_build import BuildEvidenceError, canonical, digest, file_identity
from .link_provenance import clean_toolchain_environment


def _provider(kind, **data):
    body = {'kind': kind, **data}
    return {'provider_id': digest(body), **body}


def compiler_header_roots(text):
    match = re.search(r'#include <\.\.\.> search starts here:\n(.*?)'
                      r'End of search list\.', text, re.S)
    if not match:
        raise BuildEvidenceError('missing independent compiler header search list')
    roots = []
    for line in match.group(1).splitlines():
        name = line.strip().removesuffix(' (framework directory)')
        path = Path(name)
        if not path.is_absolute() or not path.is_dir():
            raise BuildEvidenceError('unresolved builtin compiler header root')
        roots.append(str(path.resolve(strict=True)))
    if not roots:
        raise BuildEvidenceError('empty compiler builtin header roots')
    return sorted(set(roots))


def windows_installation_roots(driver):
    """Derive VC from actual executable and SDK from machine registration."""
    import winreg

    vc = next((p for p in Path(driver).parents
               if (p / 'include' / 'vcruntime.h').is_file()
               and (p / 'lib' / 'x64').is_dir()), None)
    if vc is None:
        raise BuildEvidenceError('actual compiler has no installed VC header root')
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        r'SOFTWARE\Microsoft\Windows Kits\Installed Roots') as key:
        sdk = Path(winreg.QueryValueEx(key, 'KitsRoot10')[0]).resolve(strict=True)
    versions = sorted(p for p in (sdk / 'Include').iterdir()
                      if (p / 'ucrt' / 'corecrt.h').is_file())
    if not versions:
        raise BuildEvidenceError('registered Windows SDK has no UCRT headers')
    headers = [vc / 'include']
    libraries = [vc / 'lib' / 'x64']
    for version in versions:
        headers.extend(p for p in version.iterdir() if p.is_dir())
        libraries.extend(p for p in (sdk / 'Lib' / version.name).glob('*/x64')
                         if p.is_dir())
    return {'vc': str(vc), 'sdk': str(sdk),
            'headers': sorted(str(p.resolve()) for p in headers),
            'libraries': sorted(str(p.resolve()) for p in libraries)}


def collect_input_providers(driver, family, *, cwd, env, directory):
    providers, evidence = [], []
    clean = clean_toolchain_environment(env)
    clean['LC_ALL'] = 'C'
    if family == 'msvc':
        installed = windows_installation_roots(driver)
        providers.append(_provider('compiler_builtin_header',
                                   roots=installed['headers'],
                                   compiler=file_identity(driver),
                                   installation=installed))
    else:
        # No candidate -I/-include/--sysroot and no environment include roots.
        # Apple uses xcrun's installed SDK, never a caller-supplied SDKROOT.
        sdk_query = None
        if family == 'clang':
            sdk_argv = ['/usr/bin/xcrun', '--sdk', 'macosx', '--show-sdk-path']
            sdk = subprocess.run(sdk_argv, cwd=cwd, env=clean, text=True,
                                 capture_output=True)
            if sdk.returncode or not Path(sdk.stdout.strip()).is_dir():
                raise BuildEvidenceError('cannot resolve installed Apple SDK')
            if (env.get('SDKROOT') and Path(env['SDKROOT']).resolve() !=
                    Path(sdk.stdout.strip()).resolve()):
                raise BuildEvidenceError(
                    'actual Apple SDK differs from installed provider')
            clean['SDKROOT'] = sdk.stdout.strip()
            sdk_query = {'argv': sdk_argv, 'stdout': sdk.stdout,
                         'stderr': sdk.stderr, 'exit_code': sdk.returncode,
                         'executable': file_identity('/usr/bin/xcrun')}
        argv = [str(driver), '-E', '-x', 'c++', '-v', '-']
        query = subprocess.run(argv, input='', cwd=cwd, env=clean, text=True,
                               capture_output=True)
        if query.returncode:
            raise BuildEvidenceError('independent compiler include query failed')
        data = {'argv': argv, 'environment': clean,
                'stdout': query.stdout, 'stderr': query.stderr,
                'exit_code': query.returncode, 'sdk_query': sdk_query}
        path = directory / 'builtin-header-query.json'
        path.write_bytes(canonical(data) + b'\n')
        evidence.append(path)
        providers.append(_provider(
            'compiler_builtin_header', roots=compiler_header_roots(query.stderr),
            compiler=file_identity(driver), query=data))

    roots = sorted({str(Path(sysconfig.get_path(k)).resolve(strict=True))
                    for k in ('include', 'platinclude')})
    providers.append(_provider('python_header', roots=roots,
                               interpreter=file_identity(sys.executable),
                               version=sys.version,
                               configuration=file_identity(sysconfig.__file__)))
    try:
        distribution = importlib.metadata.distribution('pybind11')
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        files, metadata = [], []
        for item in distribution.files or []:
            path = Path(distribution.locate_file(item)).resolve()
            if str(item).endswith(('.dist-info/RECORD', '.dist-info/METADATA')):
                metadata.append(file_identity(path))
            if '/include/pybind11/' not in item.as_posix():
                continue
            if not item.hash or item.hash.mode != 'sha256':
                raise BuildEvidenceError(
                    'pybind header lacks installed distribution hash')
            identity = file_identity(path)
            expected = base64.urlsafe_b64encode(
                bytes.fromhex(identity['sha256'])).decode().rstrip('=')
            if expected != item.hash.value:
                raise BuildEvidenceError(
                    'pybind header differs from installed distribution')
            files.append(identity)
        if not files or not metadata:
            raise BuildEvidenceError(
                'incomplete installed pybind distribution inventory')
        providers.append(_provider('pybind11_distribution',
                                   version=distribution.version,
                                   files=sorted(files, key=lambda p: p['path']),
                                   metadata=metadata))
    return providers, evidence


def classify_external_inputs(providers, dependencies, source_root):
    from .effective_build import _verify_identity

    claims = []
    for provider in providers:
        body = {k: v for k, v in provider.items() if k != 'provider_id'}
        if provider.get('provider_id') != digest(body):
            raise BuildEvidenceError('changed external input provider')
        for key in ('compiler', 'interpreter', 'configuration'):
            if key in provider:
                _verify_identity(provider[key], 'header provider')
        for item in provider.get('metadata', []) + provider.get('files', []):
            _verify_identity(item, 'installed header provider inventory')
    for item in dependencies:
        path = Path(item['path'])
        if path.is_relative_to(source_root):
            continue
        eligible = []
        for provider in providers:
            if (any(path.is_relative_to(root) for root in provider.get('roots', []))
                    or item in provider.get('files', [])):
                eligible.append(provider)
        if not eligible:
            raise BuildEvidenceError(
                'unapproved external source dependency: ' + str(path))
        precedence = {'pybind11_distribution': 0, 'python_header': 1,
                      'compiler_builtin_header': 2}
        provider = min(eligible,
                       key=lambda p: (precedence[p['kind']], p['provider_id']))
        claims.append({'identity': item, 'provider_id': provider['provider_id'],
                       'provenance_kind': provider['kind']})
    return claims
