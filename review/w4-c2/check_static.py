"""Check native build links and unchanged documentation content for C2 review."""

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(text)

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
    failures = []
    articles = []
    for p, page in pages.items():
        for kind, value in page.links:
            url = urlsplit(value)
            if url.scheme or url.netloc:
                continue
            target = ((root / url.path.removeprefix('/pyvoro2/'))
                      if url.path.startswith('/pyvoro2/')
                      else p.parent / unquote(url.path)) if url.path else p
            if target.is_dir():
                target /= 'index.html'
            target = target.resolve()
            if not target.exists():
                failures.append(f'{p.relative_to(root)}: missing {value}')
            elif (kind == 'href' and url.fragment and target in pages
                  and unquote(url.fragment) not in pages[target].ids):
                failures.append(f'{p.relative_to(root)}: missing anchor {value}')
        relative = p.relative_to(root)
        old = args.baseline / relative
        if str(relative) not in ('index.html', '404.html') and old.exists():
            article = r'<article\b[\s\S]*?</article>'
            identical = (re.findall(article, p.read_text()) ==
                         re.findall(article, old.read_text()))
            articles.append({'page': str(relative), 'identical': identical})
            if not identical:
                failures.append(f'Changed article: {relative}')
    css = Path('docs/stylesheets/themes.css').read_text()
    slate = re.search(r'\[data-md-color-scheme="slate"\] \{([^}]+)', css)[1]
    bootstrap = re.search(r'body:not\(\[data-md-color-media\]\) \{([^}]+)', css)[1]
    declarations = lambda block: dict(re.findall(r'(--[\w-]+):\s*([^;]+);', block))
    assert declarations(slate) == declarations(bootstrap), 'Cold Auto tokens drifted'
    light_assets = {}
    for name in ('triclinic-voronoi', 'separator-cycle'):
        asset = f'docs/assets/landing/{name}.png'
        old = subprocess.check_output(['git', 'show', f'a9d5f3d:{asset}'])
        current = Path(asset).read_bytes()
        assert old == current
        light_assets[name] = hashlib.sha256(current).hexdigest()
    result = {'cold_auto_tokens_match_slate': True, 'html_pages': len(pages), 'failures': failures,
              'unchanged_articles': articles, 'unchanged_light_assets': light_assets}
    Path('review/w4-c2/static-results.json').write_text(json.dumps(result, indent=2)+'\n')
    assert not failures, '\n'.join(failures)
    print(f'{len(pages)} HTML pages: links pass; {len(articles)} articles unchanged')


if __name__ == '__main__':
    main()
