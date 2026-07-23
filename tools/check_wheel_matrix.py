#!/usr/bin/env python3
"""Validate the complete supported pyvoro2 release-artifact matrix."""

from __future__ import annotations

import argparse
from email.message import Message
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from typing import NamedTuple
import zipfile


EXPECTED_PROJECT_NAME = 'pyvoro2'
EXPECTED_REQUIRES_PYTHON = '>=3.10'
EXPECTED_PYTHON_TAGS = ('cp310', 'cp311', 'cp312', 'cp313', 'cp314')
EXPECTED_PLATFORMS = (
    'manylinux-x86_64',
    'windows-amd64',
    'macos-arm64',
    'macos-x86_64',
)
EXPECTED_CONTRACTS = frozenset(
    (python_tag, platform)
    for python_tag in EXPECTED_PYTHON_TAGS
    for platform in EXPECTED_PLATFORMS
)
EXPECTED_WHEEL_COUNT = len(EXPECTED_CONTRACTS)
EXPECTED_RUNTIME_REQUIREMENTS = frozenset(
    {
        ('numpy', ('<2', '>=1.23'), 'python_version < "3.11"'),
        ('numpy', ('<3', '>=1.23'), 'python_version >= "3.11"'),
    }
)

_MANYLINUX_X86_64 = re.compile(
    r'manylinux(?:1|2010|2014|_[0-9]+_[0-9]+)_x86_64'
)
_MACOS_ARM64 = re.compile(r'macosx_[0-9]+_[0-9]+_arm64')
_MACOS_X86_64 = re.compile(r'macosx_[0-9]+_[0-9]+_x86_64')
_PROJECT_NORMALIZATION = re.compile(r'[-_.]+')
_REQUIREMENT_HEAD = re.compile(
    r'^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(.*)$'
)
_VERSION_SPECIFIER = re.compile(r'(?:===|==|!=|~=|>=|<=|>|<).+')
_EXTRA_REQUIREMENT_MARKER = re.compile(r'(?:^|[^A-Za-z0-9_])extra\s*==')


class WheelMatrixError(RuntimeError):
    """Raised when release artifacts do not match the supported matrix."""


class WheelFilename(NamedTuple):
    """Parsed wheel filename fields and their supported platform contract."""

    distribution: str
    version: str
    python_tag: str
    abi_tag: str
    platform_tag: str
    platform_contract: str

    @property
    def contract(self) -> tuple[str, str]:
        """Return the interpreter/platform contract represented by the wheel."""

        return self.python_tag, self.platform_contract


class WheelMatrixSummary(NamedTuple):
    """Identity and counts for one validated release-artifact directory."""

    project_name: str
    version: str
    wheel_count: int
    sdist_count: int


def normalize_project_name(name: str) -> str:
    """Return the comparison form used for project distribution names."""

    return _PROJECT_NORMALIZATION.sub('-', name).lower()


def classify_platform_tag(platform_tag: str) -> str:
    """Map one supported wheel platform tag to its release contract."""

    tags = platform_tag.split('.')
    if not tags or any(not tag for tag in tags):
        raise WheelMatrixError(f'invalid empty platform tag in {platform_tag!r}')
    if len(tags) != len(set(tags)):
        raise WheelMatrixError(
            f'duplicate compressed platform tags in {platform_tag!r}'
        )

    if all(_MANYLINUX_X86_64.fullmatch(tag) for tag in tags):
        return 'manylinux-x86_64'
    if tags == ['win_amd64']:
        return 'windows-amd64'
    if all(_MACOS_ARM64.fullmatch(tag) for tag in tags):
        return 'macos-arm64'
    if all(_MACOS_X86_64.fullmatch(tag) for tag in tags):
        return 'macos-x86_64'

    raise WheelMatrixError(
        f'unsupported platform tag {platform_tag!r}; expected only manylinux '
        'x86_64, win_amd64, macOS arm64, or macOS x86_64'
    )


def parse_wheel_filename(path: Path) -> WheelFilename:
    """Parse and validate one supported wheel filename."""

    if path.suffix != '.whl':
        raise WheelMatrixError(f'{path.name} is not a wheel filename')

    parts = path.name[:-4].split('-')
    if len(parts) != 5:
        raise WheelMatrixError(
            f'{path.name} does not have the expected project-version-'
            'python-abi-platform wheel filename form'
        )
    distribution, version, python_tag, abi_tag, platform_tag = parts
    if not distribution or not version:
        raise WheelMatrixError(
            f'{path.name} has an empty distribution name or version'
        )
    if python_tag not in EXPECTED_PYTHON_TAGS:
        raise WheelMatrixError(
            f'{path.name} has unsupported interpreter tag {python_tag!r}; '
            f'expected {", ".join(EXPECTED_PYTHON_TAGS)}'
        )
    if abi_tag != python_tag:
        raise WheelMatrixError(
            f'{path.name} has ABI tag {abi_tag!r}; expected {python_tag!r}'
        )

    return WheelFilename(
        distribution=distribution,
        version=version,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
        platform_contract=classify_platform_tag(platform_tag),
    )


def _only_archive_member(
    names: list[str],
    *,
    suffix: str,
    label: str,
) -> str:
    matches = [
        name
        for name in names
        if len(PurePosixPath(name).parts) == 2 and name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise WheelMatrixError(
            f'{label} must contain exactly one top-level {suffix} member; '
            f'found {len(matches)}'
        )
    return matches[0]


def _metadata_field(message: Message, field: str, *, label: str) -> str:
    values = message.get_all(field, [])
    if len(values) != 1 or not str(values[0]).strip():
        raise WheelMatrixError(
            f'{label} must contain exactly one nonempty {field} field'
        )
    return str(values[0]).strip()


def _parse_metadata(payload: bytes, *, label: str) -> Message:
    try:
        return BytesParser(policy=compat32).parsebytes(payload)
    except Exception as exc:
        raise WheelMatrixError(f'could not parse metadata in {label}: {exc}') from exc


def _canonical_runtime_requirement(
    value: str,
    *,
    label: str,
) -> tuple[str, tuple[str, ...], str]:
    requirement, separator, marker = value.partition(';')
    match = _REQUIREMENT_HEAD.fullmatch(requirement)
    if match is None:
        raise WheelMatrixError(
            f'{label} has invalid Requires-Dist entry {value!r}'
        )

    name = normalize_project_name(match.group(1))
    specifier_text = match.group(2).strip()
    if specifier_text.startswith('(') and specifier_text.endswith(')'):
        specifier_text = specifier_text[1:-1].strip()

    specifiers: list[str] = []
    if specifier_text:
        for item in specifier_text.split(','):
            specifier = re.sub(r'\s+', '', item)
            if not specifier or _VERSION_SPECIFIER.fullmatch(specifier) is None:
                raise WheelMatrixError(
                    f'{label} has unsupported Requires-Dist entry {value!r}'
                )
            specifiers.append(specifier)
    if len(specifiers) != len(set(specifiers)):
        raise WheelMatrixError(
            f'{label} has duplicate version clauses in Requires-Dist {value!r}'
        )

    normalized_marker = ' '.join(marker.split()) if separator else ''
    return name, tuple(sorted(specifiers)), normalized_marker


def _format_runtime_requirement(
    contract: tuple[str, tuple[str, ...], str],
) -> str:
    name, specifiers, marker = contract
    requirement = name + ','.join(specifiers)
    if marker:
        requirement += f'; {marker}'
    return requirement


def _assert_runtime_metadata(message: Message, *, label: str) -> None:
    requires_python = _metadata_field(
        message,
        'Requires-Python',
        label=label,
    )
    if re.sub(r'\s+', '', requires_python) != EXPECTED_REQUIRES_PYTHON:
        raise WheelMatrixError(
            f'{label} has Requires-Python {requires_python!r}; '
            f'expected {EXPECTED_REQUIRES_PYTHON!r}'
        )

    runtime_requirements: list[tuple[str, tuple[str, ...], str]] = []
    for raw_value in message.get_all('Requires-Dist', []):
        value = str(raw_value).strip()
        _, separator, marker = value.partition(';')
        normalized_marker = ' '.join(marker.split()) if separator else ''
        if _EXTRA_REQUIREMENT_MARKER.search(normalized_marker):
            continue
        runtime_requirements.append(
            _canonical_runtime_requirement(value, label=label)
        )

    actual = frozenset(runtime_requirements)
    if (
        len(runtime_requirements) != len(actual)
        or actual != EXPECTED_RUNTIME_REQUIREMENTS
    ):
        actual_text = ', '.join(
            _format_runtime_requirement(item) for item in sorted(actual)
        )
        expected_text = ', '.join(
            _format_runtime_requirement(item)
            for item in sorted(EXPECTED_RUNTIME_REQUIREMENTS)
        )
        raise WheelMatrixError(
            f'{label} runtime Requires-Dist entries [{actual_text}] do not '
            f'match expected entries [{expected_text}]'
        )


def _assert_project_identity(
    message: Message,
    *,
    expected_version: str,
    label: str,
) -> None:
    name = _metadata_field(message, 'Name', label=label)
    version = _metadata_field(message, 'Version', label=label)
    if normalize_project_name(name) != EXPECTED_PROJECT_NAME:
        raise WheelMatrixError(
            f'{label} has project name {name!r}; expected {EXPECTED_PROJECT_NAME!r}'
        )
    if version != expected_version:
        raise WheelMatrixError(
            f'{label} has version {version!r}; expected {expected_version!r}'
        )


def _native_module_members(names: list[str], module_name: str) -> list[str]:
    prefix = f'{module_name}.'
    return [
        name
        for name in names
        if PurePosixPath(name).parent == PurePosixPath(EXPECTED_PROJECT_NAME)
        and PurePosixPath(name).name.startswith(prefix)
        and PurePosixPath(name).suffix in {'.so', '.pyd'}
    ]


def _expanded_filename_tags(filename: WheelFilename) -> set[str]:
    return {
        f'{python_tag}-{abi_tag}-{platform_tag}'
        for python_tag in filename.python_tag.split('.')
        for abi_tag in filename.abi_tag.split('.')
        for platform_tag in filename.platform_tag.split('.')
    }


def check_wheel(path: Path) -> WheelFilename:
    """Validate filename, metadata, tags, and native modules for one wheel."""

    filename = parse_wheel_filename(path)
    if normalize_project_name(filename.distribution) != EXPECTED_PROJECT_NAME:
        raise WheelMatrixError(
            f'{path.name} has distribution {filename.distribution!r}; '
            f'expected {EXPECTED_PROJECT_NAME!r}'
        )

    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            metadata_member = _only_archive_member(
                names,
                suffix='.dist-info/METADATA',
                label=path.name,
            )
            wheel_member = _only_archive_member(
                names,
                suffix='.dist-info/WHEEL',
                label=path.name,
            )
            metadata = _parse_metadata(
                zf.read(metadata_member),
                label=f'{path.name}:{metadata_member}',
            )
            wheel_metadata = _parse_metadata(
                zf.read(wheel_member),
                label=f'{path.name}:{wheel_member}',
            )
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise WheelMatrixError(f'could not inspect wheel {path.name}: {exc}') from exc

    expected_dist_info = (
        f'{filename.distribution}-{filename.version}.dist-info'
    )
    metadata_root = PurePosixPath(metadata_member).parts[0]
    wheel_root = PurePosixPath(wheel_member).parts[0]
    if metadata_root != expected_dist_info or wheel_root != expected_dist_info:
        raise WheelMatrixError(
            f'{path.name} has inconsistent .dist-info directory naming'
        )

    _assert_project_identity(
        metadata,
        expected_version=filename.version,
        label=f'{path.name} METADATA',
    )
    _assert_runtime_metadata(
        metadata,
        label=f'{path.name} METADATA',
    )
    root_is_pure = _metadata_field(
        wheel_metadata,
        'Root-Is-Purelib',
        label=f'{path.name} WHEEL',
    )
    if root_is_pure.lower() != 'false':
        raise WheelMatrixError(
            f'{path.name} must declare Root-Is-Purelib: false'
        )

    wheel_tags = [
        str(tag).strip()
        for tag in wheel_metadata.get_all('Tag', [])
        if str(tag).strip()
    ]
    expected_tags = _expanded_filename_tags(filename)
    if len(wheel_tags) != len(set(wheel_tags)) or set(wheel_tags) != expected_tags:
        raise WheelMatrixError(
            f'{path.name} WHEEL tags {sorted(wheel_tags)!r} do not match '
            f'filename tags {sorted(expected_tags)!r}'
        )

    for module_name in ('_core', '_core2d'):
        members = _native_module_members(names, module_name)
        if not members:
            raise WheelMatrixError(
                f'{path.name} does not contain pyvoro2/{module_name}'
            )

    return filename


def check_sdist(path: Path, *, expected_version: str) -> None:
    """Validate the source-distribution filename and core metadata."""

    expected_filename = f'{EXPECTED_PROJECT_NAME}-{expected_version}.tar.gz'
    if path.name != expected_filename:
        raise WheelMatrixError(
            f'sdist filename {path.name!r} does not match {expected_filename!r}'
        )

    try:
        with tarfile.open(path, 'r:gz') as tf:
            members = [member for member in tf.getmembers() if member.isfile()]
            names = [member.name for member in members]
            pkg_info_member = _only_archive_member(
                names,
                suffix='/PKG-INFO',
                label=path.name,
            )
            extracted = tf.extractfile(pkg_info_member)
            if extracted is None:
                raise WheelMatrixError(
                    f'could not read {pkg_info_member} from {path.name}'
                )
            metadata = _parse_metadata(
                extracted.read(),
                label=f'{path.name}:{pkg_info_member}',
            )
    except (OSError, tarfile.TarError) as exc:
        raise WheelMatrixError(f'could not inspect sdist {path.name}: {exc}') from exc

    expected_root = expected_filename[:-7]
    if PurePosixPath(pkg_info_member).parts[0] != expected_root:
        raise WheelMatrixError(
            f'{path.name} has inconsistent top-level directory naming'
        )
    _assert_project_identity(
        metadata,
        expected_version=expected_version,
        label=f'{path.name} PKG-INFO',
    )
    _assert_runtime_metadata(
        metadata,
        label=f'{path.name} PKG-INFO',
    )


def _format_contract(contract: tuple[str, str]) -> str:
    return f'{contract[0]}/{contract[1]}'


def validate_wheel_matrix(dist_dir: Path) -> WheelMatrixSummary:
    """Validate exactly 20 supported wheels and one matching sdist."""

    if not dist_dir.is_dir():
        raise WheelMatrixError(f'artifact directory does not exist: {dist_dir}')

    entries = sorted(dist_dir.iterdir(), key=lambda path: path.name)
    unexpected = [
        path.name
        for path in entries
        if not path.is_file()
        or not (path.name.endswith('.whl') or path.name.endswith('.tar.gz'))
    ]
    if unexpected:
        raise WheelMatrixError(
            f'artifact directory contains unexpected entries: {", ".join(unexpected)}'
        )

    wheels = [path for path in entries if path.name.endswith('.whl')]
    sdists = [path for path in entries if path.name.endswith('.tar.gz')]
    if len(wheels) != EXPECTED_WHEEL_COUNT:
        raise WheelMatrixError(
            f'expected exactly {EXPECTED_WHEEL_COUNT} wheels, found {len(wheels)}'
        )
    if len(sdists) != 1:
        raise WheelMatrixError(
            f'expected exactly one source distribution, found {len(sdists)}'
        )

    contracts: dict[tuple[str, str], Path] = {}
    versions: set[str] = set()
    for wheel in wheels:
        filename = check_wheel(wheel)
        previous = contracts.get(filename.contract)
        if previous is not None:
            raise WheelMatrixError(
                f'duplicate wheel contract {_format_contract(filename.contract)}: '
                f'{previous.name}, {wheel.name}'
            )
        contracts[filename.contract] = wheel
        versions.add(filename.version)

    actual_contracts = set(contracts)
    if actual_contracts != EXPECTED_CONTRACTS:
        missing = sorted(EXPECTED_CONTRACTS - actual_contracts)
        unsupported = sorted(actual_contracts - EXPECTED_CONTRACTS)
        details: list[str] = []
        if missing:
            details.append(
                'missing ' + ', '.join(_format_contract(item) for item in missing)
            )
        if unsupported:
            details.append(
                'unsupported '
                + ', '.join(_format_contract(item) for item in unsupported)
            )
        raise WheelMatrixError('wheel contract mismatch: ' + '; '.join(details))

    if len(versions) != 1:
        raise WheelMatrixError(
            'wheel versions are inconsistent: ' + ', '.join(sorted(versions))
        )
    version = next(iter(versions))
    check_sdist(sdists[0], expected_version=version)

    return WheelMatrixSummary(
        project_name=EXPECTED_PROJECT_NAME,
        version=version,
        wheel_count=len(wheels),
        sdist_count=len(sdists),
    )


def main() -> int:
    """Validate a downloaded release-artifact directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dist_dir',
        type=Path,
        nargs='?',
        default=Path('dist'),
        help='directory containing the merged wheels and sdist',
    )
    args = parser.parse_args()

    summary = validate_wheel_matrix(args.dist_dir)
    print(
        f'validated {summary.project_name} {summary.version}: '
        f'{summary.wheel_count} wheels and {summary.sdist_count} sdist'
    )
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except WheelMatrixError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
