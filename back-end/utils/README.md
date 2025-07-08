# UN Document Crawler & Parser

This repository provides two modular Python scripts designed to support automated crawling and parsing of United Nations (UN) documents, especially PDF-based resolutions from the official UN document portal.

## Overview

1. **`crawler.py`** – A Selenium-based web crawler to download UN documents matching specific search criteria. Metadata for each document is saved alongside the downloads.

2. **`parser.py`** – A document parser that extracts structured content from the downloaded PDFs (e.g., Markdown/JSON), optionally supporting OCR and layout analysis for complex documents.

---

## Prerequisites

- Python 3.10+
- ChromeDriver (you can download the latest ChromeDriver from [here](https://googlechromelabs.github.io/chrome-for-testing/)). Your ChromeDriver must match with your Chrome browser.
- OCR capabilities enabled (e.g., Tesseract or GPU-accelerated OCR via `docling`)
- Virtual environment recommended

## Usage

Configure in __main__ block and run `python crawler.py` and `python parser.py`



