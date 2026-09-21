"""Exercise native provenance allocation and graph validation without Python."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CASES = (
    'occurrence_tokens',
    'deep_assignment',
    'full_cycle_validation',
    'construction_bounds',
    'token_exhaustion',
    'higher_order_and_marginal',
    'memory_relocation',
    'uncontracted_power_primitives',
    'uncontracted_standard_plane',
)


def _cmake() -> str:
    try:
        import cmake
    except ImportError:
        executable = shutil.which('cmake')
    else:
        # The repository's cmake/ directory can be a namespace package when
        # the PyPI CMake package is absent; then use the ordinary PATH.
        executable = shutil.which('cmake', path=getattr(cmake, 'CMAKE_BIN_DIR', None))
    if executable is None:
        pytest.skip('CMake is required to compile the standalone native tests')
    return executable


def _run(command: list[str], env: dict[str, str]) -> str:
    completed = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        timeout=180,
    )
    assert completed.returncode == 0, (
        f'{command!r} exited {completed.returncode}\n'
        f'{completed.stdout}\n{completed.stderr}'
    )
    print(completed.stdout, end='')
    return completed.stdout


@pytest.fixture(scope='session')
def native_witness_executable(tmp_path_factory) -> Path:
    env = os.environ.copy()
    # An unavailable shell-specific compiler must not prevent CMake from
    # discovering the toolchain used by normal Linux/macOS/Windows CI.
    for variable in ('CC', 'CXX'):
        compiler = env.get(variable)
        if compiler and not shutil.which(compiler):
            command = shlex.split(compiler, posix=os.name != 'nt')
            if not command or not shutil.which(command[0].strip('"')):
                env.pop(variable)
    build = tmp_path_factory.mktemp('native-witness-build')
    cmake = _cmake()
    sanitize = env.get('PYVORO2_NATIVE_TEST_SANITIZERS') == '1'
    _run([
        cmake, '-S', str(REPO_ROOT / 'tests' / 'native'), '-B', str(build),
        '-DCMAKE_BUILD_TYPE=Release',
        f'-DPYVORO2_NATIVE_TEST_SANITIZERS={"ON" if sanitize else "OFF"}',
    ], env)
    _run([cmake, '--build', str(build), '--config', 'Release', '--verbose'], env)
    basename = 'test_native_witness.exe' if os.name == 'nt' else (
        'test_native_witness'
    )
    candidates = (build / basename, build / 'Release' / basename)
    for executable in candidates:
        if executable.is_file():
            return executable
    pytest.fail(f'CMake did not produce the native test executable: {build}')


@pytest.mark.parametrize('case', CASES)
def test_native_witness_cpp(native_witness_executable: Path, case: str) -> None:
    output = _run(
        [str(native_witness_executable), case], os.environ.copy(),
    )
    assert f'native witness: {case} passed' in output
