# Final Code with Dynamic Thresholding without Average Distances and added colors for bubbles
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import imutils
import string
import os
import tempfile
from typing import Dict
from collections import OrderedDict
import threading, time
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_JSON_DIR = os.path.join(tempfile.gettempdir(), "omr-json")
os.makedirs(TEMP_JSON_DIR, exist_ok=True)

# === Config ===
MIN_AREA = 250
FILL_THRESHOLD = 0.8


# ------------------ Utility Functions ------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [-15, -15],
        [maxWidth + 10, -10],
        [maxWidth + 10, maxHeight + 10],
        [-10, maxHeight + 10]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def crop_sheet(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return four_point_transform(image, approx.reshape(4, 2))
    return image


# ------------------ Bubble Detection ------------------
def separate_circles_and_squares(cnts):
    circles, squares = [], []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        circularity = (4 * np.pi * area) / (peri ** 2 + 1e-6)
        if len(approx) == 4:
            squares.append(c)
        elif circularity >= 0.85 or len(approx) > 4:
            circles.append(c)
        else:
            squares.append(c)
    return circles, squares


def compute_dynamic_thresholds(square_coords):
    row_y_thresh = 20
    col_x_thresh = 20

    if len(square_coords) > 1:
        xs = sorted([s["cx"] for s in square_coords])
        ys = sorted([s["cy"] for s in square_coords])

        if len(xs) > 1:
            avg_x_gap = np.mean(np.diff(xs))
            col_x_thresh = max(10, int(avg_x_gap * 0.4))

        if len(ys) > 1:
            avg_y_gap = np.mean(np.diff(ys))
            row_y_thresh = max(10, int(avg_y_gap * 0.4))

    return row_y_thresh, col_x_thresh


def group_squares_into_columns(square_coords, col_x_thresh):
    square_coords.sort(key=lambda s: s["cx"])
    cols = []
    current_col = [square_coords[0]]
    for sq in square_coords[1:]:
        if abs(sq["cx"] - current_col[-1]["cx"]) < col_x_thresh:
            current_col.append(sq)
        else:
            current_col.sort(key=lambda s: s["cy"])
            cols.append(current_col)
            current_col = [sq]
    current_col.sort(key=lambda s: s["cy"])
    cols.append(current_col)
    return cols


def analyze_sheet(image: np.ndarray, debug_dir: str) -> Dict:
    orig = image.copy()
    labeled_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # --- save threshold debug ---
    thresh_path = os.path.join(debug_dir, "threshold.png")
    cv2.imwrite(thresh_path, thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    circles, squares = separate_circles_and_squares(cnts)

    square_coords = []
    for s in squares:
        x, y, w, h = cv2.boundingRect(s)
        square_coords.append({"x": x, "y": y, "w": w, "h": h, "cx": x + w // 2, "cy": y + h // 2})
        cv2.rectangle(orig, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # ✅ Compute dynamic thresholds
    row_thresh, col_thresh = compute_dynamic_thresholds(square_coords)
    cols = group_squares_into_columns(square_coords, col_thresh)

    circle_points = []
    for c in circles:
        (x, y), radius = cv2.minEnclosingCircle(c)
        cx, cy, r = int(x), int(y), int(radius)
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.circle(mask, (cx, cy), r, 255, -1)

        total_pixels = cv2.countNonZero(mask)
        white_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
        fill_ratio = white_pixels / (total_pixels + 1e-6)
        filled = fill_ratio >= FILL_THRESHOLD

        # ✅ Draw green for filled, red for unfilled
        color = (0, 255, 0) if filled else (0, 0, 255)
        cv2.circle(orig, (cx, cy), r, color, 2)
        cv2.circle(labeled_img, (cx, cy), r, color, 2)

        circle_points.append({"x": cx, "y": cy, "r": r, "fill_ratio": fill_ratio, "filled": filled})

    letters = list(string.ascii_uppercase)
    filled_answers = {}
    q_counter = 0

    for col_idx, col in enumerate(cols, start=1):
        for sq in col:
            row_circles = [c for c in circle_points if abs(c["y"] - sq["cy"]) < row_thresh]
            sorted_circles = sorted([c for c in row_circles if c["x"] < sq["cx"]],
                                    key=lambda c: c["x"], reverse=True)
            if len(sorted_circles) < 4:
                continue

            selected = sorted(sorted_circles[:4], key=lambda c: c["x"])

            q_counter += 1
            new_label = f"Q{q_counter}"
            filled_options = []

            cv2.putText(labeled_img, new_label, (sq["x"], sq["y"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            for i, c in enumerate(selected):
                cv2.putText(labeled_img, letters[i], (c["x"], c["y"]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                if c["filled"]:
                    filled_options.append({
                        "option": letters[i],
                        "fill_ratio": round(c["fill_ratio"], 2)
                    })

            filled_answers[new_label] = filled_options if filled_options else None

    annotated_path = os.path.join(debug_dir, "annotated.png")
    cv2.imwrite(annotated_path, orig)

    labeled_path = os.path.join(debug_dir, "annotated_labeled.png")
    cv2.imwrite(labeled_path, labeled_img)

    return OrderedDict(sorted(filled_answers.items(), key=lambda x: int(x[0][1:])))


# ------------------ FastAPI Endpoints ------------------
@app.post("/process/{temp_name}")
async def process_sheet(file: UploadFile = File(...), temp_name: str = "result"):
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, file.filename)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    image = cv2.imread(input_path)
    if image is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    cropped = crop_sheet(image)
    result = analyze_sheet(cropped, tmp_dir)
    link = os.path.join(tmp_dir, "annotated_labeled.png")

    json_path = os.path.join(TEMP_JSON_DIR, f"{temp_name}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)

    return FileResponse(link)


@app.get("/get_json/{file_path:path}")
async def get_json(file_path: str):
    json_path = os.path.join(TEMP_JSON_DIR, f"{file_path}.json")
    if not os.path.exists(json_path):
        return JSONResponse({"error": "File not found or expired"}, status_code=404)
    return FileResponse(json_path)


def cleanup_old_json_files():
    while True:
        now = time.time()
        for filename in os.listdir(TEMP_JSON_DIR):
            fpath = os.path.join(TEMP_JSON_DIR, filename)
            if now - os.path.getmtime(fpath) > 3600:
                os.remove(fpath)
        time.sleep(600)


@app.on_event("startup")
def start_cleanup():
    threading.Thread(target=cleanup_old_json_files, daemon=True).start()
