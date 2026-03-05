import os
from pathlib import Path
from PyPDF2 import PdfReader

KB = Path("knowledge_base")

def pdf_to_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    out = []
    for p in reader.pages:
        out.append(p.extract_text() or "")
    return "\n".join(out)

def main():
    if not KB.exists():
        raise SystemExit("knowledge_base folder not found")

    for folder in KB.iterdir():
        if not folder.is_dir():
            continue
        for pdf in folder.rglob("*.pdf"):
            txt = pdf.with_suffix(".txt")
            if txt.exists():
                continue  

            try:
                text = pdf_to_text(pdf)
                txt.write_text(text, encoding="utf-8", errors="ignore")
                print(f"Wrote {txt}")
            except Exception as e:
                print(f"Failed {pdf}: {e}")

if __name__ == "__main__":
    main()