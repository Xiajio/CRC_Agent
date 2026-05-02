from pathlib import Path

from pypdf import PdfReader

from scripts.generate_project_intro_pdf import generate_pdf


def test_generate_project_intro_pdf() -> None:
    output_dir = Path("tmp/pdfs/test-artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "智能体产品介绍.pdf"

    generated_path = generate_pdf(output_path)

    assert generated_path.exists()
    assert generated_path.stat().st_size > 50_000

    reader = PdfReader(str(generated_path))
    assert len(reader.pages) >= 8
    assert "智能体产品介绍" in (reader.metadata or {}).get("/Title", "")

    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    for anchor in [
        "Product Overview",
        "Product Architecture",
        "Core Modules",
        "Future Roadmap",
        "Application Scenarios",
        "Technology Highlights",
        "Data Security",
        "Summary & Outlook",
    ]:
        assert anchor in extracted
