"""Build Korean slides and a complete line-numbered source appendix; no browser needed."""
import argparse
import ast
import hashlib
import html
import importlib.util
import json
import math
import re
from pathlib import Path
import shutil
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def escape(text):
    return html.escape(str(text), quote=True)


def wrap_line(text, width):
    if len(text) <= width:
        return [text]
    indent = len(text) - len(text.lstrip())
    return textwrap.wrap(text, width=width, subsequent_indent=" " * min(indent + 4, 24),
                         replace_whitespace=False, drop_whitespace=False,
                         break_long_words=True, break_on_hyphens=False) or [""]


def source_lines(path, symbol=None):
    text = path.read_text()
    lines = text.splitlines()
    if not symbol:
        return list(enumerate(lines, 1))
    current = ast.parse(text)
    for part in symbol.split("."):
        current = next(node for node in current.body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == part)
    return list(enumerate(lines[current.lineno - 1:current.end_lineno], current.lineno))


def code_html(lines, width=114):
    if not lines:
        return ''
    rows = []
    for number, line in lines:
        for i, segment in enumerate(wrap_line(line, width)):
            label = str(number) if i == 0 else "↳"
            rows.append(f'<div class="code-line"><span class="line-number">{label}</span>'
                        f'<span class="line-text">{highlight(segment) or " "}</span></div>')
    return '<div class="code">' + ''.join(rows) + '</div>'


def highlight(line):
    pattern = r'''(\#.*$|"[^"\n]*"|'[^'\n]*'|\b(?:def|class|return|import|from|if|else|elif|for|in|while|with|as|raise|try|except|finally|not|and|or|is|None|True|False|yield)\b|\b\d+(?:\.\d+)?\b)'''
    result, position = [], 0
    for match in re.finditer(pattern, line):
        result.append(escape(line[position:match.start()]))
        token = match.group()
        kind = 'comment' if token.startswith('#') else ('str' if token.startswith(('"', "'")) else ('num' if token[0].isdigit() else 'kw'))
        result.append(f'<span class="{kind}">{escape(token)}</span>')
        position = match.end()
    return ''.join(result) + escape(line[position:])


def chunks(lines, width=114, rows=27):
    block, count = [], 0
    for item in lines:
        cost = len(wrap_line(item[1], width))
        if block and count + cost > rows:
            yield block
            block, count = [], 0
        block.append(item)
        count += cost
    if block:
        yield block


def diagram(labels, chapter):
    """Original editable SVG teaching schematic, not a reproduced paper figure."""
    width, height, gap = 1160, 150, 20
    box = (width - gap * (len(labels) - 1)) / len(labels)
    items = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 150">']
    for i, label in enumerate(labels):
        x = i * (box + gap)
        items.append(f'<rect x="{x}" y="30" width="{box}" height="85" rx="4" fill="#e5eff2" stroke="#94b9bd"/>')
        # Short words are split around slash only if necessary; no clipped label.
        words = textwrap.wrap(label, width=18) or [label]
        for j, word in enumerate(words):
            y = 68 + j * 23 - (len(words) - 1) * 10
            items.append(f'<text x="{x + box / 2}" y="{y}" text-anchor="middle" font-size="17" fill="#173d53" font-family="Pretendard,sans-serif">{escape(word)}</text>')
        if i < len(labels) - 1:
            right = x + box
            items.append(f'<path d="M {right + 3} 72 L {right + gap - 4} 72 M {right + gap - 9} 67 L {right + gap - 4} 72 L {right + gap - 9} 77" stroke="#007e83" stroke-width="2" fill="none"/>')
    items.append('</svg>')
    return '<div class="diagram">' + ''.join(items) + '</div>'


def scientific_figure(kind):
    """Signal/sequence teaching figures generated from analytical examples."""
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 150">']
    if kind == 1:
        for row, freq, label in ((0, 3, "continuous sine"), (1, 11, "discrete samples")):
            cy = 38 + row * 70
            points = [(180 + j * 2, cy - 24 * math.sin(2 * math.pi * freq * j / 440)) for j in range(441)]
            parts.append(f'<text x="5" y="{cy}" font-size="17" fill="#173d53">{label}</text>')
            parts.append('<polyline points="' + ' '.join(f'{x:.2f},{y:.2f}' for x, y in points) + '" fill="none" stroke="#007e83" stroke-width="2"/>')
            if row:
                for x, y in points[::16]:
                    parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="#cf784d"/>')
    elif kind == 5:
        for i, labels in enumerate((('가','가','blank','가'), ('가','blank','가','가'))):
            for j, label in enumerate(labels):
                x, y = 100 + j * 200, 12 + i * 65
                color = '#f2e7db' if label == 'blank' else '#e1eff0'
                parts.append(f'<rect x="{x}" y="{y}" width="160" height="48" fill="{color}"/>')
                parts.append(f'<text x="{x+80}" y="{y+30}" text-anchor="middle" font-size="23">{label}</text>')
            parts.append(f'<text x="930" y="{44+i*65}" font-size="23" fill="#007e83">→ 가가</text>')
    elif kind == 12:
        for i, token in enumerate(('h₁','h₂','h₃')):
            parts.append(f'<text x="{100+i*175}" y="35" font-size="24">{token}</text>')
        for i, label in enumerate(('h₁','h₁','h₂','h₃','h₃','h₃')):
            color = ('#d1e8e8','#d1e8e8','#e5dce8','#ede0cb','#ede0cb','#ede0cb')[i]
            x = 80+i*170
            parts.append(f'<rect x="{x}" y="76" width="140" height="50" fill="{color}"/><text x="{x+70}" y="109" text-anchor="middle" font-size="23">{label}</text>')
        parts.append('<text x="680" y="35" font-size="22">duration = [2, 1, 3]</text>')
    elif kind == 19:
        for i in range(3):
            points = [(90+i*250+j, 130-i*35-30*(.5-.5*math.cos(2*math.pi*j/380))) for j in range(381)]
            parts.append('<polyline points="' + ' '.join(f'{x:.2f},{y:.2f}' for x,y in points) + '" fill="none" stroke="#007e83" stroke-width="3"/>')
            parts.append(f'<text x="{90+i*250}" y="{85-i*35}" font-size="18">chunk {i+1}</text>')
    else:
        return ''
    return '<div class="diagram">' + ''.join(parts) + '</svg></div>'


def build(limit=None):
    spec = importlib.util.spec_from_file_location("course", ROOT / "lecture/course.py")
    course = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(course)
    chapters = course.CHAPTERS if limit is None else course.CHAPTERS[:limit]
    figures = json.loads((ROOT / 'lecture/figures.json').read_text())
    pages = []
    def page(title, body, label="STE SPEECH · 한국어", css=""):
        number = len(pages) + 1
        pages.append(f'<section class="slide {css}" id="page-{number}"><div class="eyebrow">{label}</div>'
                     f'<h1>{escape(title)}</h1>{body}<div class="footer">만들면서 배우는 음성 생성 AI · 2026-11-20'
                     f'<span class="page-number">{number}</span></div></section>')
    page("만들면서 배우는 음성 생성 AI", '<div class="chapter-mark">SteSpeech</div>'
         '<p class="lead">한국어 음성 인식 · 한국어 음성 합성 · 음악 음원 분리</p>'
         '<p class="lead">이론과 논문, 모델 설계, 전체 실행 코드, 학습 및 평가 절차</p>'
         '<p class="small">KsponSpeech · KSS · MUSDB18-HQ<br>학습·데이터셋 평가 미실행 / 서버 67 합성 입력 구조 검증</p>', css="cover")
    for group in (chapters[:12], chapters[12:]):
        if group:
            page("강의 구성", '<table><tr><th>장</th><th>주제</th><th>주요 코드</th></tr>' + ''.join(
                f'<tr><td>{c["number"]:02d}</td><td>{escape(c["title"])}</td><td>{escape(c["source"])}</td></tr>' for c in group) + '</table>', css="toc")
    page("실행 범위와 필수 자료", '<table><tr><th>항목</th><th>현재 상태</th></tr>'
        '<tr><td>모델·전처리·학습·평가·추론 코드</td><td>전체 소스 포함. 자동 다운로드 없음</td></tr>'
        '<tr><td>구조 검사</td><td>67번 RTX 5090에서 합성 입력으로 검증</td></tr>'
        '<tr><td>실제 공개 예제 그림</td><td>음성 각 1발화, 음악 1곡의 7초 구간만 전처리·시각화</td></tr>'
        '<tr><td>훈련·데이터셋 평가</td><td>미실행. 성능 수치·학습 curve 없음</td></tr>'
        '<tr><td>KSS duration</td><td>실제 자모 토큰 정렬 자료 필요. 가짜 duration으로 대체하지 않음</td></tr>'
        '<tr><td>Inference</td><td>실제 사용에는 학습된 checkpoint 필요. TTS에는 vocoder도 필요</td></tr>'
        '<tr><td>논문 재현 범위</td><td>교육용 변형. 원 구조와 다른 점을 해당 장에 명시</td></tr></table>')
    for c in chapters:
        n, title = c["number"], c["title"]
        refs = ' · '.join(f'<a href="{course.REFERENCES[r][1]}">{escape(r)}</a>' for r in c["refs"])
        page(title, f'<div class="chapter-mark">CHAPTER {n:02d}</div><p class="lead">{escape(c["goal"])}</p>'
             + diagram(c["diagram"], n) + f'<p class="small">코드: {escape(c["source"])} · {escape(c["symbol"])}<br>참고 자료: {refs}</p>',
             label=f'{n:02d} / 학습 목표', css="cover")
        theory = ''.join(f'<h2>{escape(heading)}</h2><p>{escape(body)}</p>' for heading, body in c["theory"])
        page(title + " · 핵심 이론", theory, label=f'{n:02d} / 개념과 가정')
        for key, visual in figures.items():
            if visual['chapter'] != n:
                continue
            notes = ''.join(f'<h2>{i+1:02d} / 읽는 법</h2><p>{escape(note)}</p>'
                            for i, note in enumerate(visual['notes']))
            media = {'kss_speech': ['kss'], 'kspon_speech': ['kspon'],
                     'music_stems': ['music_mixture']}
            audio = ''.join(f'<p class="audio-preview"><audio controls preload="none" src="media/{name}.wav"></audio></p>'
                            for name in media.get(key, []))
            body = '<div class="visual-layout"><div class="visual-main">'
            body += f'<img src="{visual["file"]}" alt="{escape(visual["title"])}"></div>'
            body += f'<aside class="visual-notes">{notes}{audio}</aside></div>'
            body += f'<p class="visual-source">{escape(visual["kind"])} · {escape(visual["source"])}</p>'
            page(visual['title'], body, label=f'{n:02d} / 데이터와 모델을 그림으로 읽기', css='visual')
        figure = scientific_figure(n) or diagram(c["diagram"], n)
        page(title + " · 연산과 차원", figure + f'<div class="formula">{escape(c["formula"])}</div>'
             + ''.join(f'<p>{escape(detail)}</p>' for detail in c["detail"]), label=f'{n:02d} / 저자 작성 설명 그림')
        selected = source_lines(ROOT / c["source"], c["symbol"])
        excerpt = next(chunks(selected, width=87, rows=22))
        heading = f'{c["source"]} · {excerpt[0][0]}–{excerpt[-1][0]}행'
        body = '<div class="two-col"><div class="code-col"><div class="code-meta">' + escape(heading) + '</div>' + code_html(excerpt,87) + '</div><div class="notes">'
        body += ''.join(f'<h2>{i+1}</h2><p>{escape(note)}</p>' for i, note in enumerate(c["notes"]))
        body += '</div></div><p class="small">긴 함수의 나머지 줄과 호출 경로는 전체 코드 부록에서 확인합니다.</p>'
        page(title + " · 실제 코드", body, label=f'{n:02d} / source와 동일한 줄번호')
        bibliography = ''.join(f'<p><a href="{course.REFERENCES[r][1]}">{escape(course.REFERENCES[r][0])}</a></p>' for r in c["refs"])
        page(title + " · 확인 문제와 읽기", f'<h2>확인 문제</h2><p class="lead">{escape(c["exercise"])}</p>'
             f'<div class="answer"><h2>해설</h2><p>{escape(c["answer"])}</p></div><h2>연결해서 읽을 자료</h2><div class="references">{bibliography}</div>',
             label=f'{n:02d} / 복습과 출처')
    if limit is None:
        page('그림과 공개 예제의 출처',
             '<p>그림용으로만 받은 공개 발화·7초 음악은 lecture/media에 둡니다. 학습 manifest에 넣거나 실제 모델 평가에 사용하지 않았습니다.</p>'
             '<table class="visual-index"><tr><th>자료</th><th>확인 가능한 출처와 사용 범위</th></tr>'
             '<tr><td>KSS</td><td>Kyubyong Park (2018). Bingsu/KSS_Dataset의 train row 0 한 발화. CC BY-NC-SA 4.0. 파형·STFT·mel·F0·energy와 그 변형 그림도 해당 조건을 따릅니다.</td></tr>'
             '<tr><td>KsponSpeech</td><td>Bang et al. (2020). 저자 공개 jubang0219/KsponSpeech, eval_clean row 0 한 발화. 전처리 시각화만 수행.</td></tr>'
             '<tr><td>음악 preview</td><td>Stöter, Liutkus &amp; Ito (2018), SiSEC18-MUS 7s Excerpts, Zenodo 1256064. A Classic Education — NightOwl. mixture와 네 정답 stem의 7초 WAV. 학습 대상 MUSDB18-HQ와 preview release를 구별합니다.</td></tr>'
             '<tr><td>직접 그린 모델</td><td>원 논문과 code/를 대조해 작성했습니다. 실제 구현과 원 논문의 차이를 각 페이지에 표시합니다.</td></tr></table>'
             '<p class="small">파일 SHA256·샘플 수·주파수·변환 조건: lecture/media/provenance.json. 그림 생성 코드: scripts/build_figures.py.</p>', label='출처 / 재현 가능한 그림')
        page('원 자료로 연결하기',
             '<p><a href="https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset">KSS — Kyubyong Park, 원 데이터</a><br>'
             '<a href="https://huggingface.co/datasets/Bingsu/KSS_Dataset">사용한 공개 예제 mirror — Bingsu/KSS_Dataset</a></p>'
             '<p><a href="https://doi.org/10.3390/app10196936">KsponSpeech — Bang et al., Applied Sciences (2020)</a><br>'
             '<a href="https://huggingface.co/datasets/jubang0219/KsponSpeech">사용한 저자 공개 예제 — jubang0219/KsponSpeech</a></p>'
             '<p><a href="https://doi.org/10.5281/zenodo.1256064">SiSEC18-MUS 7s Excerpts — Stöter, Liutkus &amp; Ito (2018)</a><br>'
             '<a href="https://sigsep.github.io/datasets/musdb.html">MUSDB18 / MUSDB18-HQ — SigSep 공식 설명</a></p>'
             '<p>그래프의 실제 측정값, 수학적 예제, oracle mask, 구조 개념도를 각 페이지에서 구별합니다. 예제 audio는 HTML에서만 재생하며 PDF에는 대응하는 파형과 출처를 수록합니다.</p>', label='참고 자료 / 데이터 예제')
        runbook = (ROOT / "lecture/runbook.json").read_text() if (ROOT / "lecture/runbook.json").exists() else "[]"
        for entry in json.loads(runbook):
            page(entry["title"], ''.join(f'<p>{escape(p)}</p>' for p in entry.get("paragraphs", []))
                 + code_html(list(enumerate(entry.get("code", []), 1)), 110), label="실습 절차 / 실행 여부 명시")
        sources = sorted((ROOT / "code").glob("*.py")) + [ROOT / "scripts/ste_speech.py", ROOT / "scripts/verify.py"]
        digest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        page("전체 실행 코드", '<p class="lead">본문에서 설명한 코드의 전체 소스를 수록합니다.</p>'
             '<p>각 페이지에 원 파일 경로와 실제 줄번호를 표시합니다. 긴 줄의 ↳ 표시는 같은 원본 줄의 시각적 줄바꿈입니다.</p>'
             '<p>학습과 평가 코드는 TODO가 아닌 실행 함수입니다. 아래 코드를 수록하는 과정에서는 학습·데이터셋 평가를 호출하지 않았습니다.</p>', css="cover")
        for path in sources:
            relative = str(path.relative_to(ROOT))
            blocks = list(chunks(source_lines(path), rows=28))
            # Keep a one-line tail from becoming a nearly empty appendix page.
            if len(blocks) > 1 and sum(len(wrap_line(t,114)) for _,t in blocks[-1]) < 8:
                while sum(len(wrap_line(t,114)) for _,t in blocks[-1]) < 10:
                    blocks[-1].insert(0, blocks[-2].pop())
            for block in blocks:
                page(relative, f'<div class="code-meta">원본 {block[0][0]}–{block[-1][0]}행 · SHA256 {digest[relative][:16]}</div>'
                     + code_html(block), label="전체 소스 / 생략 없음", css="appendix")
        (ROOT / "lecture/source-integrity.json").write_text(json.dumps(digest, indent=2))
    output = ROOT / "lecture/SteSpeech-KO.html"
    output.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>만들면서 배우는 음성 생성 AI</title>'
                     '<link rel="stylesheet" href="style.css"><body>' + ''.join(pages) + '</body></html>', encoding="utf-8")
    return output, len(pages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-only", action="store_true")
    parser.add_argument("--preview-chapters", type=int)
    args = parser.parse_args()
    source, count = build(args.preview_chapters)
    print(json.dumps({"html": str(source), "slides": count}, ensure_ascii=False))
    if not args.html_only:
        executable = shutil.which("weasyprint")
        if not executable:
            raise SystemExit("PDF 생성에는 WeasyPrint가 필요합니다. HTML 생성은 완료했습니다.")
        subprocess.run([executable, str(source), str(source.with_suffix(".pdf"))], check=True)
