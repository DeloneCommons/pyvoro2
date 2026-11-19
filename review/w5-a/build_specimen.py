"""Build a review-only specimen with native Zensical, outside production docs.

Usage: python review/w5-a/build_specimen.py /absolute/fresh/work-directory
The output site is WORK/site; nothing is added to production navigation.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[2]
work = Path(sys.argv[1]).resolve()
work.mkdir(parents=True, exist_ok=False)
shutil.copytree(root / 'docs', work / 'docs')
shutil.copytree(root / 'overrides', work / 'overrides')
# The changelog uses a repository-root snippet.
shutil.copy2(root / 'CHANGELOG.md', work / 'CHANGELOG.md')
(work / 'docs/specimen').mkdir()
shutil.copy2(Path(__file__).with_name('specimen.md'), work / 'docs/specimen/index.md')
config = yaml.safe_load((root / 'mkdocs.yml').read_text())
config['nav'].append({'Review specimen': 'specimen/index.md'})
(work / 'mkdocs.yml').write_text(yaml.safe_dump(config, sort_keys=False))
subprocess.run([sys.executable, '-m', 'zensical', 'build', '--clean', '--strict'],
               cwd=work, check=True)
print(work / 'site')
