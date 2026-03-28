from pathlib import Path
import re
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas


def main() -> None:
    base = Path('/Users/ruthvikbandari/Desktop/Diafoot CV')
    md_path = base / 'Midterm_Report_DiaFootAI_v2.md'
    pdf_path = base / 'Midterm_Report_DiaFootAI_v2.pdf'

    text = md_path.read_text(encoding='utf-8')

    lines = text.splitlines()
    clean: list[str] = []
    for ln in lines:
        s = ln.rstrip()
        s = re.sub(r'^#{1,6}\s*', '', s)
        s = re.sub(r'^\s*[-*]\s+', '• ', s)
        s = s.replace('**', '')
        s = s.replace('`', '')
        if s.strip() == '---':
            s = ''
        clean.append(s)

    c = canvas.Canvas(str(pdf_path), pagesize=LETTER)
    width, height = LETTER
    left_margin = 54
    right_margin = 54
    top_margin = 54
    bottom_margin = 54
    line_height = 14
    max_width = width - left_margin - right_margin

    font_name = 'Helvetica'
    font_size = 11
    c.setFont(font_name, font_size)

    y = height - top_margin

    for para in clean:
        if para.strip() == '':
            y -= line_height
            if y < bottom_margin:
                c.showPage()
                c.setFont(font_name, font_size)
                y = height - top_margin
            continue

        words = para.split()
        current = ''
        wrapped: list[str] = []
        for w in words:
            trial = (current + ' ' + w).strip()
            if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
                current = trial
            else:
                if current:
                    wrapped.append(current)
                current = w
        if current:
            wrapped.append(current)

        for line in wrapped:
            c.drawString(left_margin, y, line)
            y -= line_height
            if y < bottom_margin:
                c.showPage()
                c.setFont(font_name, font_size)
                y = height - top_margin

    c.save()
    print(f'PDF created: {pdf_path}')


if __name__ == '__main__':
    main()
