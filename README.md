# Dynamic OMR Evaluation API

A FastAPI-based Optical Mark Recognition (OMR) system that automatically detects and evaluates marked answers from OMR sheets without requiring predefined templates.

This project uses **OpenCV-based image processing** to dynamically detect bubbles, identify filled options, and generate structured results.

---

# Features

* Dynamic OMR detection (no fixed template required)
* Supports multiple image formats
* Supports PDF uploads (automatically converts first page to image)
* Automatic OMR sheet cropping and perspective correction
* Circle and square detection using contour analysis
* Bubble fill detection using pixel ratio
* Multi-column question detection
* Annotated debug image output
* JSON result storage
* Automatic cleanup of temporary files
* FastAPI REST API interface

---

# Supported File Types

The API accepts the following file formats:

* JPG
* JPEG
* PNG
* BMP
* TIFF
* PDF (first page processed automatically)

---

# Technology Stack

* **FastAPI** – API framework
* **OpenCV** – Image processing
* **NumPy** – Numerical computation
* **Imutils** – OpenCV utilities
* **pdf2image** – Convert PDF to images
* **Python** – Core programming language

---

# Project Structure

```
OMR-API/
│
├── MyOMR.py            # Main FastAPI application
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
│
├── inputs/             # Sample input OMR sheets
├── outputs/            # Generated outputs
```

---

# Installation

### 1. Clone the repository

```
git clone https://github.com/BSampathKumar2004/OMR_Evalution.git
cd OMR_Evalution
```

---

### 2. Create a virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

Windows

```
venv\Scripts\activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Install system dependency (for PDF support)

Linux:

```
sudo apt install poppler-utils
```

---

# Running the API

Start the FastAPI server:

```
uvicorn FinalOMR:app --reload
```

Server will run at:

```
http://127.0.0.1:8000
```

Swagger API documentation:

```
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## Process OMR Sheet

Upload an OMR sheet and process it.

```
POST /process/{temp_name}
```

Example:

```
POST /process/test_sheet
```

Input:

* file: OMR sheet image or PDF

Response:

* Annotated OMR image showing detected bubbles.

---

## Get JSON Result

Retrieve the detected answers in JSON format.

```
GET /get_json/{temp_name}
```

Example:

```
GET /get_json/test_sheet
```

Example Response:

```
{
  "Q1": [{"option": "B", "fill_ratio": 0.92}],
  "Q2": [{"option": "C", "fill_ratio": 0.88}],
  "Q3": null
}
```

---

# OMR Detection Pipeline

The system processes the OMR sheet using the following steps:

1. Image upload
2. PDF conversion (if needed)
3. OMR sheet boundary detection
4. Perspective transformation
5. Grayscale conversion
6. Binary thresholding (Otsu)
7. Contour detection
8. Circle and square classification
9. Question grouping
10. Bubble fill ratio calculation
11. Answer extraction
12. JSON output generation

---

# Example Output

Annotated image showing:

* Green bubbles → Filled answers
* Red bubbles → Unfilled options
* Blue boxes → Question markers
* Labels → Question numbers and options

---

# Future Improvements

Possible enhancements:

* Batch OMR processing
* Multi-page PDF support
* Support for different OMR layouts
* GPU acceleration
* Automatic sheet alignment improvement
* Cloud deployment

---

# License

This project is open-source.

---

# Author

Sampath Kumar
Backend Developer – Python & FastAPI
