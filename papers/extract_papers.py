"""
Extract text and images from each PDF into its own folder.
Output per paper:
  <paper_name>/
    text.txt          -- full extracted text, page-separated
    images/           -- every embedded raster image
    figures/          -- page-level renders of pages that contain images (for context)
"""

import fitz  # pymupdf
import os
import sys
from pathlib import Path

PAPERS_DIR = Path(__file__).parent
MIN_IMAGE_SIZE = 100  # skip tiny icons (px in either dimension)

def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")

def extract_paper(pdf_path: Path):
    stem = pdf_path.stem
    out_dir = PAPERS_DIR / stem
    img_dir = out_dir / "images"
    fig_dir = out_dir / "figures"

    out_dir.mkdir(exist_ok=True)
    img_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    doc = fitz.open(pdf_path)
    text_parts = []
    img_count = 0
    pages_with_images = set()

    # --- pass 1: extract embedded images ---
    for page_num, page in enumerate(doc, start=1):
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            w, h = base_image["width"], base_image["height"]
            if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
                continue  # skip tiny decorative elements
            ext = base_image["ext"]
            img_bytes = base_image["image"]
            img_filename = img_dir / f"p{page_num:03d}_img{img_index+1:02d}.{ext}"
            img_filename.write_bytes(img_bytes)
            img_count += 1
            pages_with_images.add(page_num)

    # --- pass 2: render every page at high DPI ---
    # Architecture diagrams and other vector figures won't appear as embedded images,
    # so we render all pages to ensure nothing is missed.
    mat = fitz.Matrix(2.0, 2.0)  # 144 dpi
    for page_num in range(1, len(doc) + 1):
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=mat)
        fig_path = fig_dir / f"page_{page_num:03d}.png"
        pix.save(str(fig_path))

    # --- pass 3: extract text ---
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text_parts.append(f"{'='*60}\nPage {page_num}\n{'='*60}\n{text}")

    text_out = out_dir / "text.txt"
    text_out.write_text("\n\n".join(text_parts), encoding="utf-8")

    page_count = doc.page_count
    doc.close()
    print(f"  [{stem}]  pages={page_count}  images={img_count}  figure-pages={len(pages_with_images)}")

def main():
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(pdfs)} papers in {PAPERS_DIR}\n")
    for pdf in pdfs:
        extract_paper(pdf)
    print("\nDone.")

if __name__ == "__main__":
    main()
