import json
import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

class DocumentParser:
    def __init__(self, num_threads=8):
        """
        Initialize the DocumentParser with default options.
        """
        self.num_threads = num_threads

    def parse_pdf(self, pdf_path, device=None):
        """
        Parse a PDF or JSON file and extract its content.

        Args:
            pdf_path (str): Path to the PDF or JSON file.
            device (str, optional): Device to use for processing ('cpu' or 'cuda').
                                    If None, it will automatically detect GPU availability.

        Returns:
            str: Extracted content from the document.
        """
        # Automatically determine the device if not specified
        print(f"Using device: {device}")
        print("Starting PDF parsing...")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if pdf_path.endswith(".json"):
            return self._parse_json(pdf_path)
        elif pdf_path.endswith(".pdf"):
            return self._parse_pdf_with_pipeline(pdf_path, device)

    def _parse_json(self, json_path):
        """
        Parse a JSON file and extract paragraphs and headings.

        Args:
            json_path (str): Path to the JSON file.

        Returns:
            str: Extracted content from the JSON file.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            parsed_json = json.load(f)
        paragraphs = [item["text"] for item in parsed_json if item["type"] in ["paragraph", "heading"]]
        return "\n".join(paragraphs)

    def _parse_pdf_with_pipeline(self, pdf_path, device):
        """
        Parse a PDF file using the document pipeline.

        Args:
            pdf_path (str): Path to the PDF file.
            device (str): Device to use for processing ('cpu' or 'cuda').

        Returns:
            str: Extracted content from the PDF file.
        """
        if device == 'cpu':
            accelerator_options = AcceleratorOptions(num_threads=self.num_threads, device=AcceleratorDevice.CPU)
        else:
            accelerator_options = AcceleratorOptions(num_threads=self.num_threads, device=AcceleratorDevice.CUDA)

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
                InputFormat.ASCIIDOC,
                InputFormat.MD,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        converted = converter.convert(pdf_path)
        return converted.document.export_to_markdown()