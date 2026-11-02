#!/usr/bin/env python3
"""
Drone Labeller - a local web tool for boxing drones in frame folders.

Pointing at an 8 px drone in words does not work. This serves frame folders in a browser so the
box can be drawn where the drone actually is, and writes the YOLO label straight back to disk
beside the image.

Three things here exist because of what these frames are:

  contrast on demand   CLAHE, served as a query parameter. Many hard frames have the drone in
                       shade against a building where the region averages 52/255 and nothing is
                       visible raw. The enhancement is for LOOKING ONLY - the stored image on
                       disk is never modified.
  snap                 a click is turned into a box that hugs whatever is actually there, so
                       labels do not end up with a lot of empty space around the target.
  assisted candidates  a detection model plus frame differencing plus a local z-score, offered
                       as suggestions to accept or ignore. The model is never trusted on its
                       own - it is usually the model being retrained.

Expected folder layout, which is left exactly as it is found:

    <root>/<folder>/images/*.jpg
                   /labels/*.txt        YOLO, class 0, created when missing
                   /annotated/*.jpg     optional, kept in step if present
                   /crops/*.jpg         optional, kept in step if present
                   /REVIEW.csv          optional, kept in step if present

Usage:
  ./server.py --root /path/to/labelled/folders
  ./server.py --root ... --port 8001 --model /path/to/weights.pt
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
app = FastAPI(title="Drone Labeller")

ROOT: Path = Path(".")
MODEL_PATH: str | None = None
_model = None
_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
FIELDS = ["image", "labelled", "score", "box_w_px", "found_by", "note"]


# ---------------------------------------------------------------- helpers

def folder_dir(folder: str) -> Path:
    d = (ROOT / folder).resolve()
    root = ROOT.resolve()
    if d != root and root not in d.parents:
        raise HTTPException(400, "folder outside root")
    if not (d / "images").is_dir():
        raise HTTPException(404, f"no images/ in {folder}")
    return d


def frame_no(name: str) -> int:
    """
    The trailing run of digits only.

    Joining every digit in the name gives 400001 for "fixed outdoor vid-4 dji mavic_00001" -
    the 4 from "vid-4" leaks in and the ordering silently changes.
    """
    m = re.findall(r"(\d+)", Path(name).stem)
    return int(m[-1]) if m else 0


def read_boxes(lp: Path, w: int, h: int):
    if not lp.exists():
        return []
    out = []
    for line in lp.read_text().splitlines():
        t = line.split()
        if len(t) != 5:
            continue
        cx, cy, bw, bh = (float(v) for v in t[1:])
        out.append([round((cx - bw / 2) * w, 1), round((cy - bh / 2) * h, 1),
                    round((cx + bw / 2) * w, 1), round((cy + bh / 2) * h, 1)])
    return out


def write_boxes(lp: Path, boxes, w: int, h: int) -> int:
    lines = []
    for b in boxes:
        x0, x1 = sorted((max(0.0, float(b[0])), min(float(w), float(b[2]))))
        y0, y1 = sorted((max(0.0, float(b[1])), min(float(h), float(b[3]))))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        lines.append("0 %.6f %.6f %.6f %.6f"
                     % ((x0+x1)/2/w, (y0+y1)/2/h, (x1-x0)/w, (y1-y0)/h))
    lp.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def redraw(d: Path, name: str, boxes):
    """Keep annotated/ and crops/ in step, but only if the folder already has them."""
    if not (d / "annotated").is_dir() and not (d / "crops").is_dir():
        return
    im = cv2.imread(str(d / "images" / name))
    if im is None:
        return
    h, w = im.shape[:2]
    for b in boxes:
        cv2.rectangle(im, (int(b[0])-2, int(b[1])-2), (int(b[2])+2, int(b[3])+2), (0, 0, 255), 1)
    if (d / "annotated").is_dir():
        cv2.imwrite(str(d / "annotated" / name), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if (d / "crops").is_dir():
        C = 640
        if boxes:
            cx = sum((b[0] + b[2]) / 2 for b in boxes) / len(boxes)
            cy = sum((b[1] + b[3]) / 2 for b in boxes) / len(boxes)
        else:
            cx, cy = w / 2, h / 2
        X0 = int(max(0, min(cx - C/2, w - C))); Y0 = int(max(0, min(cy - C/2, h - C)))
        crop = np.full((C, C, 3), 114, np.uint8)
        piece = im[Y0:Y0+C, X0:X0+C]
        crop[:piece.shape[0], :piece.shape[1]] = piece
        cv2.imwrite(str(d / "crops" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])


def update_review(d: Path, name: str, n_boxes: int, width: float, note: str):
    f = d / "REVIEW.csv"
    if not f.exists():
        return
    rows = list(csv.DictReader(open(f, encoding="utf-8-sig")))
    hit = False
    for r in rows:
        if r.get("image") == name:
            hit = True
            r.update(labelled="yes" if n_boxes else "no", score="",
                     box_w_px=int(width), found_by="labelled by hand", note=note)
    if not hit:
        rows.append(dict(image=name, labelled="yes" if n_boxes else "no", score="",
                         box_w_px=int(width), found_by="labelled by hand", note=note))
    with open(f, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows([{k: r.get(k, "") for k in FIELDS} for r in rows])


# ---------------------------------------------------------------- routes

@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "static" / "index.html").read_text()


@app.get("/api/folders")
def folders():
    out = []
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        if (d / "images").is_dir():
            out.append({"name": d.name, "count": len(list((d / "images").glob("*.jpg")))})
    return out


@app.get("/api/frames")
def frames(folder: str):
    d = folder_dir(folder)
    notes = {}
    f = d / "REVIEW.csv"
    if f.exists():
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            notes[r.get("image", "")] = r.get("note", "")
    out = []
    for p in sorted((d / "images").glob("*.jpg"), key=lambda q: frame_no(q.name)):
        lp = d / "labels" / (p.stem + ".txt")
        n = 0
        if lp.exists():
            n = sum(1 for line in lp.read_text().splitlines() if len(line.split()) == 5)
        out.append({"name": p.name, "n": frame_no(p.name), "boxes": n,
                    "has_label": lp.exists(), "note": notes.get(p.name, "")})
    return out


@app.get("/api/image")
def image(folder: str, name: str, enhance: int = 0, clip: float = 4.0):
    d = folder_dir(folder)
    p = d / "images" / name
    if not p.exists():
        raise HTTPException(404, "no such frame")
    if not enhance:
        return FileResponse(p, media_type="image/jpeg")
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    cl = cv2.createCLAHE(clipLimit=max(1.0, clip), tileGridSize=(8, 8))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(cl.apply(g), cv2.COLOR_GRAY2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.get("/api/label")
def get_label(folder: str, name: str):
    d = folder_dir(folder)
    im = cv2.imread(str(d / "images" / name))
    if im is None:
        raise HTTPException(404, "no such frame")
    h, w = im.shape[:2]
    lp = d / "labels" / (Path(name).stem + ".txt")
    return {"w": int(w), "h": int(h), "boxes": read_boxes(lp, w, h), "has_label": lp.exists()}


class SaveReq(BaseModel):
    folder: str
    name: str
    boxes: list
    note: str = ""


@app.post("/api/label")
def save_label(req: SaveReq):
    d = folder_dir(req.folder)
    im = cv2.imread(str(d / "images" / req.name))
    if im is None:
        raise HTTPException(404, "no such frame")
    h, w = im.shape[:2]
    lp = d / "labels" / (Path(req.name).stem + ".txt")
    lp.parent.mkdir(exist_ok=True)
    n = write_boxes(lp, req.boxes, w, h)
    widest = max((b[2] - b[0] for b in req.boxes), default=0)
    redraw(d, req.name, req.boxes)
    update_review(d, req.name, n, widest,
                  req.note or ("" if n else "no drone in frame - marked by hand"))
    return {"ok": True, "boxes": n}


class DelReq(BaseModel):
    folder: str
    name: str


@app.post("/api/delete")
def delete_frame(req: DelReq):
    d = folder_dir(req.folder)
    stem = Path(req.name).stem
    removed = []
    for sub, ext in (("images", ".jpg"), ("labels", ".txt"), ("annotated", ".jpg"),
                     ("crops", ".jpg")):
        p = d / sub / (stem + ext)
        if p.exists():
            p.unlink(); removed.append(sub)
    f = d / "REVIEW.csv"
    if f.exists():
        rows = [r for r in csv.DictReader(open(f, encoding="utf-8-sig"))
                if r.get("image") != req.name]
        with open(f, "w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows([{k: r.get(k, "") for k in FIELDS} for r in rows])
        removed.append("REVIEW.csv row")
    return {"ok": True, "removed": removed}


class AiReq(BaseModel):
    folder: str
    name: str


@app.post("/api/ai")
def ai(req: AiReq):
    """Candidates from three independent sources. Suggestions only - nothing is written."""
    global _model
    d = folder_dir(req.folder)
    ps = sorted((d / "images").glob("*.jpg"), key=lambda q: frame_no(q.name))
    names = [p.name for p in ps]
    if req.name not in names:
        raise HTTPException(404, "no such frame")
    i = names.index(req.name)
    g = cv2.imread(str(ps[i]), cv2.IMREAD_GRAYSCALE)
    H, W = g.shape
    out, notes = [], []

    if MODEL_PATH:
        try:
            if _model is None:
                from ultralytics import YOLO
                import torch
                _model = (YOLO(MODEL_PATH), 0 if torch.cuda.is_available() else "cpu")
            m, dev = _model
            r = m.predict(str(ps[i]), imgsz=1280, conf=0.0005, iou=0.5,
                          verbose=False, device=dev)[0]
            if r.boxes is not None and len(r.boxes):
                for xy, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                    out.append({"box": [round(float(v), 1) for v in xy], "src": "model",
                                "score": round(float(c), 4)})
        except Exception as e:
            notes.append(f"model unavailable: {type(e).__name__}")
    else:
        notes.append("no model configured (--model)")

    if 0 < i < len(ps) - 1:
        a = cv2.imread(str(ps[i-1]), cv2.IMREAD_GRAYSCALE).astype(np.int16)
        b = cv2.imread(str(ps[i+1]), cv2.IMREAD_GRAYSCALE).astype(np.int16)
        dd = np.minimum(a - g, b - g)
        dd = cv2.GaussianBlur(np.clip(dd, 0, 255).astype(np.uint8), (3, 3), 0)
        thr = max(6, int(np.percentile(dd, 99.9)))
        nn, _, st, _ = cv2.connectedComponentsWithStats((dd > thr).astype(np.uint8), 8)
        for k in range(1, nn):
            x, y, w, h, area = (int(v) for v in st[k])
            if 3 <= max(w, h) <= 260 and area >= 4:
                out.append({"box": [x, y, x + w, y + h], "src": "motion",
                            "score": round(float(dd[y:y+h, x:x+w].max()) / 255, 3)})

    e = _clahe.apply(g).astype(np.float32)
    bg = cv2.GaussianBlur(e, (0, 0), 11)
    sq = cv2.GaussianBlur(e * e, (0, 0), 11)
    sd = np.sqrt(np.maximum(sq - bg * bg, 1.0))
    for z, tag in (((bg - e) / sd, "dark"), ((e - bg) / sd, "bright")):
        thr = max(3.5, float(np.percentile(z, 99.95)))
        nn, _, st, _ = cv2.connectedComponentsWithStats((z > thr).astype(np.uint8), 8)
        for k in range(1, nn):
            x, y, w, h, area = (int(v) for v in st[k])
            if 3 <= max(w, h) <= 260 and area >= 6 and max(w, h) / max(1, min(w, h)) <= 6:
                out.append({"box": [x, y, x + w, y + h], "src": tag,
                            "score": round(float(z[y:y+h, x:x+w].max()) / 20, 3)})

    out = [o for o in out if o["box"]]
    out.sort(key=lambda o: -o["score"])
    return {"w": int(W), "h": int(H), "candidates": out[:60], "notes": notes}


class SnapReq(BaseModel):
    folder: str
    name: str
    x: float
    y: float
    rad: int = 70


@app.post("/api/snap")
def snap(req: SnapReq):
    """Turn a click into a box that hugs whatever is there, so labels stay tight."""
    d = folder_dir(req.folder)
    g = cv2.imread(str(d / "images" / req.name), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise HTTPException(404, "no such frame")
    H, W = g.shape
    sx, sy, rad = int(req.x), int(req.y), max(18, int(req.rad))
    x0, y0 = max(0, sx - rad), max(0, sy - rad)
    x1, y1 = min(W, sx + rad), min(H, sy + rad)
    e = _clahe.apply(g[y0:y1, x0:x1]).astype(np.float32)
    bg = cv2.GaussianBlur(e, (0, 0), 9)
    sq = cv2.GaussianBlur(e * e, (0, 0), 9)
    sd = np.sqrt(np.maximum(sq - bg * bg, 1.0))
    best = None
    for z in ((bg - e) / sd, (e - bg) / sd):
        for q in (99.0, 97.0, 94.0, 90.0):
            m = cv2.morphologyEx((z > float(np.percentile(z, q))).astype(np.uint8),
                                 cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            nn, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
            for i in range(1, nn):
                x, y, w, h, a = (int(v) for v in st[i])
                if max(w, h) < 3 or max(w, h) > 2 * rad or a < 4:
                    continue
                dist = float(np.hypot(x0 + x + w/2 - sx, y0 + y + h/2 - sy))
                if dist > rad * 0.8:
                    continue
                sc = a / (1 + dist / 12.0)
                if best is None or sc > best[0]:
                    best = (sc, x0 + x, y0 + y, w, h)
            if best:
                break
        if best:
            break
    if best is None:
        return {"box": None}
    _, x, y, w, h = (float(v) for v in best)
    return {"box": [x, y, x + w, y + h]}


def main():
    global ROOT, MODEL_PATH
    ap = argparse.ArgumentParser(description="Local web labeller for drone frame folders.")
    ap.add_argument("--root", type=Path, required=True,
                    help="directory containing the frame folders")
    ap.add_argument("--model", default=None,
                    help="optional YOLO weights, enables assisted candidates")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    ROOT = args.root
    MODEL_PATH = args.model
    if not ROOT.is_dir():
        sys.exit(f"no such directory: {ROOT}")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    import uvicorn
    print(f"root   {ROOT}")
    print(f"model  {MODEL_PATH or 'none - assisted candidates limited to imaging'}")
    print(f"open   http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
