from pathlib import Path

from backend.api.services.upload_fixture_cards import load_fixture_upload_card


def test_demo_upload_fixture_files_are_paired():
    upload_root = Path("tests/fixtures/demo_uploads")
    card_root = Path("tests/fixtures/upload_cards")

    for stem in ("demo_colonoscopy_report", "demo_pathology_report"):
        assert (upload_root / f"{stem}.pdf").is_file()
        assert (card_root / f"{stem}.json").is_file()


def test_demo_upload_cards_load_by_uploaded_filename():
    colonoscopy_card = load_fixture_upload_card("demo_colonoscopy_report.pdf")
    pathology_card = load_fixture_upload_card("demo_pathology_report.pdf")

    assert colonoscopy_card["type"] == "medical_visualization_card"
    assert pathology_card["type"] == "medical_visualization_card"
