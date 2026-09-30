"""Controlled build configuration refuses generators without attribution."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('generator', [
    'Visual Studio 18 2026', 'Ninja Multi-Config', 'Unix Makefiles',
])
def test_evidence_mode_refuses_other_generators_before_source_measurement(
        tmp_path, generator):
    cmake = (shutil.which('cmake')
             or shutil.which('cmake', path=str(Path(sys.executable).parent)))
    if cmake is None:
        pytest.skip('CMake is unavailable')
    script = tmp_path / 'unsupported-generator.cmake'
    script.write_text(
        f'set(CMAKE_GENERATOR "{generator}")\n'
        f'include("{(ROOT / "cmake/NativeQualification.cmake").as_posix()}")\n',
        encoding='utf8',
    )
    environment = os.environ.copy()
    environment['PYVORO2_EVIDENCE_DIR'] = str(tmp_path / 'records')
    result = subprocess.run([cmake, '-P', str(script)], cwd=tmp_path,
                            env=environment, text=True, capture_output=True)
    assert result.returncode != 0
    assert 'Controlled qualification requires the Ninja generator' in result.stderr
    assert not (tmp_path / 'native-source-measurement.json').exists()
