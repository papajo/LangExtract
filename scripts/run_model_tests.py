import io
import json
import os
import sys
from typing import Any, Iterable, Tuple

import langextract as lx
from dotenv import load_dotenv
from langextract.core import data as lx_data


def _result_to_jsonable(result: Any) -> Any:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "__dict__"):
        return result.__dict__
    return result


def _format_table(rows: Iterable[Iterable[str]]) -> str:
    lines = []
    for row in rows:
        cells = [(cell or "").replace("\n", " ").strip() for cell in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _extract_pdf_text(data: bytes) -> str:
    try:
        import pdfplumber
    except Exception:  # noqa: BLE001
        pdfplumber = None

    if pdfplumber is not None:
        parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                parts.append(f"=== Page {page_index} ===")
                try:
                    text = page.extract_text(layout=True)
                except TypeError:
                    text = page.extract_text()
                if text:
                    parts.append(text)
                tables = page.extract_tables() or []
                for table_index, table in enumerate(tables, start=1):
                    parts.append(f"--- Table {page_index}.{table_index} ---")
                    parts.append(_format_table(table))
        return "\n".join(p for p in parts if p)

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _ocr_pdf_text(data: bytes) -> Tuple[str, str]:
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except Exception as exc:  # noqa: BLE001
        return "", f"OCR support unavailable: {exc}"

    images = convert_from_bytes(data)
    ocr_text = "\n".join(pytesseract.image_to_string(img) for img in images)
    return ocr_text, ""


def _load_pdf_text(path: str, use_ocr: bool = True) -> str:
    with open(path, "rb") as handle:
        data = handle.read()
    text = _extract_pdf_text(data)
    if text.strip() or not use_ocr:
        return text
    ocr_text, ocr_error = _ocr_pdf_text(data)
    if ocr_error:
        raise RuntimeError(ocr_error)
    return ocr_text


def _write_result(name: str, payload: dict[str, Any]) -> None:
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"{name}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_model_tests.py <pdf_path>")
        return 1

    load_dotenv()
    pdf_path = sys.argv[1]
    text = _load_pdf_text(pdf_path, use_ocr=True)

    prompt_description = (
        "Extract the key categories and bullet items from this document. "
        "Return a list of extractions with class=category and the text "
        "containing the items listed under that category."
    )
    examples = [
        lx_data.ExampleData(
            text="AI Fundamentals - PyTorch, NVIDIA GPUs.",
            extractions=[
                lx_data.Extraction(
                    extraction_class="category",
                    extraction_text="AI Fundamentals",
                )
            ],
        )
    ]

    models: list[dict[str, Any]] = [
        {
            "name": "ollama_gemma3_4b",
            "model_id": "gemma3:4b",
            "model_url": "http://localhost:11434",
            "fence_output": False,
            "use_schema_constraints": False,
            "language_model_params": {"timeout": 300},
        }
    ]

    gemini_key = os.environ.get("LANGEXTRACT_API_KEY")
    if gemini_key:
        models.append(
            {
                "name": "gemini_2_5_flash",
                "model_id": "gemini-2.5-flash",
                "api_key": gemini_key,
                "fence_output": True,
                "use_schema_constraints": True,
            }
        )

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        models.append(
            {
                "name": "openai_gpt_4o",
                "model_id": "gpt-4o",
                "api_key": openai_key,
                "fence_output": True,
                "use_schema_constraints": False,
            }
        )

    for model in models:
        name = model.pop("name")
        payload: dict[str, Any] = {"model": model}
        try:
            result = lx.extract(
                text_or_documents=text,
                prompt_description=prompt_description,
                examples=examples,
                **model,
            )
            payload["result"] = _result_to_jsonable(result)
        except Exception as exc:  # noqa: BLE001
            payload["error"] = str(exc)
        _write_result(name, payload)
        print(f"Wrote results/{name}.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
