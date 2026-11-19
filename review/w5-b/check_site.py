"""Check built links/assets and compare unmodified articles with the base build.

Usage: python review/w5-b/check_site.py CANDIDATE_SITE BASELINE_SITE
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append((key, attrs[key]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('baseline', type=Path)
    args = parser.parse_args()
    root = args.candidate.resolve()
    pages = {p: Page(p.read_text()) for p in root.rglob('*.html')}
    failures, unchanged = [], []
    for path, page in pages.items():
        for kind, value in page.links:
            url = urlsplit(value)
            if url.scheme or url.netloc:
                continue
            if url.path.startswith('/pyvoro2/'):
                target = root / unquote(url.path.removeprefix('/pyvoro2/'))
            else:
                target = path.parent / unquote(url.path) if url.path else path
            if target.is_dir():
                target /= 'index.html'
            target = target.resolve()
            if not target.exists():
                failures.append(f'{path.relative_to(root)}: missing {value}')
            elif (kind == 'href' and url.fragment and target in pages
                  and unquote(url.fragment) not in pages[target].ids):
                failures.append(f'{path.relative_to(root)}: missing anchor {value}')
        relative = path.relative_to(root)
        old = args.baseline / relative
        if str(relative) not in (
            'examples/index.html', 'guide/index.html', 'theory/index.html', '404.html'
        ) and old.exists():
            article = r'<article\b[\s\S]*?</article>'
            before = re.findall(article, old.read_text())
            after = re.findall(article, path.read_text())
            if not before or before != after:
                failures.append(f'Changed or missing article: {relative}')
            else:
                unchanged.append(str(relative))
    # Keep the incoming maintenance deep link as well as every outgoing route.
    example = root / 'examples/index.html'
    if 'how-the-published-examples-are-maintained' not in pages[example].ids:
        failures.append('Missing retained Examples maintenance anchor')
    assets = {}
    for path in (root / 'assets/landing').glob('*.png'):
        old = args.baseline / path.relative_to(root)
        if path.read_bytes() != old.read_bytes():
            failures.append(f'Changed scientific asset: {path.name}')
        assets[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(html_pages=len(pages), failures=failures,
                  unchanged_articles=unchanged, unchanged_assets=assets)
    Path(__file__).with_name('static-results.json').write_text(
        json.dumps(result, indent=2) + '\n')
    assert not failures, '\n'.join(failures)
    print(f'{len(pages)} pages: links/assets pass; {len(unchanged)} articles unchanged')


if __name__ == '__main__':
    main()
