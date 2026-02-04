# Invoice OCR - Hybrid Approach

A smart PDF and document OCR system that uses the **right tool for each page type** - combining free local OCR with powerful AI vision for optimal cost and accuracy.

## Features

- **Hybrid Processing**: Automatically detects page type and routes to the best OCR engine
- **Image Enhancement**: Upscales and enhances low-quality images before OCR
- **Handwriting Support**: Claude Vision reads handwritten text, receipts, and complex documents
- **Cost Optimized**: Uses free local OCR for text pages, AI only when needed
- **Excel Export**: Download results as formatted Excel or plain text
- **Web Interface**: Simple drag-and-drop upload interface

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         PDF UPLOAD                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PAGE TYPE DETECTION                             │
│                      (PyMuPDF)                                   │
│                                                                  │
│   Analyzes each page:                                            │
│   • Extractable text length                                      │
│   • Number of embedded images                                    │
│   • Image coverage percentage                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│       IMAGE PAGE         │    │        TEXT PAGE         │
│  (scanned, handwritten,  │    │  (digital PDF with       │
│   photos, receipts)      │    │   selectable text)       │
└──────────────────────────┘    └──────────────────────────┘
              │                               │
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   ENHANCE RESOLUTION     │    │       PaddleOCR          │
│                          │    │    (Local & FREE)        │
│  • Upscale if < 2000px   │    │                          │
│  • Auto-contrast         │    │  • No API cost           │
│  • Sharpen               │    │  • Fast processing       │
│  • Brightness adjust     │    │  • Good for clean text   │
└──────────────────────────┘    └──────────────────────────┘
              │                               │
              ▼                               │
┌──────────────────────────┐                  │
│     CLAUDE VISION        │                  │
│     (AWS Bedrock)        │                  │
│                          │                  │
│  • Reads handwriting     │                  │
│  • Understands context   │                  │
│  • Handles poor quality  │                  │
└──────────────────────────┘                  │
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     COMBINED OUTPUT                              │
│           Display in UI • Excel Export • Text Download           │
└─────────────────────────────────────────────────────────────────┘
```

## The Hybrid Approach Explained

### Problem
- **Claude Vision** is powerful but costs money per API call
- **Local OCR** is free but struggles with handwriting and poor quality images
- Sending every page to Claude is expensive and unnecessary

### Solution
Automatically detect what each page contains and use the appropriate tool:

| Page Type | Detection | OCR Method | Cost |
|-----------|-----------|------------|------|
| Digital PDF (text) | Has extractable text | PaddleOCR | **FREE** |
| Scanned document | No text, has images | Claude Vision | ~$0.001/page |
| Handwritten notes | Image-based content | Claude Vision | ~$0.001/page |
| Photos/Receipts | High image coverage | Claude Vision | ~$0.001/page |

### Page Detection Logic

```python
# Using PyMuPDF to analyze PDF structure
text_length = len(page.get_text())  # Extractable text
image_count = len(page.get_images()) # Embedded images
image_coverage = image_area / page_area  # % covered by images

# Decision
if text_length < 50 and image_count > 0:
    return "IMAGE"  # Scanned document
elif image_coverage > 50%:
    return "IMAGE"  # Image-heavy page
else:
    return "TEXT"   # Digital text PDF
```

## Installation

### Prerequisites

- Python 3.8+
- AWS Account with Bedrock access (for Claude Vision)
- Poppler (for PDF processing)

### 1. Install Poppler

**Windows:**
```bash
# Download from: https://github.com/osborn/poppler-windows/releases
# Extract and add 'bin' folder to PATH
```

**Linux:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure AWS Credentials

```bash
# Set environment variables or use AWS CLI
aws configure
```

Ensure your AWS account has access to:
- `anthropic.claude-3-haiku-20240307-v1:0`
- `anthropic.claude-3-sonnet-20240229-v1:0`

## Usage

### Start the Server

```bash
python app.py
```

Server runs at: `http://127.0.0.1:5000`

### Process Documents

1. Open the web interface
2. Upload a PDF or image file
3. Enable/disable image enhancement
4. Click "Process Document"
5. View results and download as Excel/Text

## Configuration

Edit the constants in `app.py`:

```python
# Server
APP_PORT = 5000

# AWS
AWS_REGION = "us-east-1"

# Image Processing
DPI = 300                    # PDF to image conversion quality
TARGET_RESOLUTION = 2000     # Upscale images smaller than this
JPEG_QUALITY = 95            # Output image quality

# Enhancement
CONTRAST_FACTOR = 1.3        # Contrast boost
SHARPNESS_FACTOR = 1.5       # Sharpness boost
BRIGHTNESS_FACTOR = 1.1      # Brightness boost
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Web Framework | FastAPI | API and web interface |
| PDF Processing | pdf2image, PyMuPDF | Convert and analyze PDFs |
| Image Enhancement | Pillow, OpenCV | Improve image quality |
| Local OCR | PaddleOCR | Free text extraction |
| AI Vision | Claude (AWS Bedrock) | Complex document reading |
| Excel Export | openpyxl | Generate spreadsheets |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page with upload form |
| `/upload` | POST | Process uploaded document |
| `/download/excel/{id}` | GET | Download results as Excel |
| `/download/text/{id}` | GET | Download results as text |

## Output Format

### Excel Export
- **Sheet 1: Extraction Report** - Summary with page breakdown
- **Sheet 2: Full Text** - Complete extracted text

### Page Breakdown
Each page shows:
- Page number
- Type (Image/Text)
- OCR method used (Claude/PaddleOCR)
- Resolution (original → enhanced)

## Cost Optimization

This hybrid approach significantly reduces costs:

**Example: 10-page invoice**
- 7 pages are digital text → PaddleOCR (FREE)
- 3 pages are scanned/handwritten → Claude Haiku (~$0.003)

**Traditional approach:** ~$0.01 for all 10 pages
**Hybrid approach:** ~$0.003 (70% savings)

## Project Structure

```
invoice_read/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Requirements

```
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
boto3>=1.28.0
pdf2image>=1.16.0
PyMuPDF>=1.23.0
Pillow>=10.0.0
numpy>=1.24.0
opencv-python-headless>=4.8.0
openpyxl>=3.1.0
paddlepaddle>=2.5.0
paddleocr>=2.7.0
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
