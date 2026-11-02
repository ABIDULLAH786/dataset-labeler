#!/usr/bin/env python3
"""
Drone Labeller - a local web tool for boxing small aerial targets in folders of frames.

Pointing at an 8 px drone in words does not work. This serves image folders in a browser so the
box can be drawn where the target actually is, and writes the YOLO label straight back to disk
beside the image.

Four things here exist because of what these frames are:

  contrast on demand   CLAHE, served as a query parameter. Many hard frames have the drone in
                       shade against a building where the region averages 52/255 and nothing is
                       visible raw. The enhancement is for LOOKING ONLY - the stored image on
                       disk is never modified.
  snap                 a click is turned into a box that hugs whatever is actually there, so
                       labels do not end up with a lot of empty space around the target.
  assisted candidates  a detection model plus frame differencing plus a local z-score, offered
                       as suggestions to accept or ignore. The model is never trusted on its
                       own - it is usually the model being retrained.

CLASSES are per dataset, not per tool. The class list is read from whatever the folder already
carries - data.yaml, classes.txt or notes.json - so an existing dataset opens with its own names
rather than something imposed on it, and is written back to classes.txt. Any class may be
renamed, added or reordered; reordering is reported back as a warning, because class indices are
positional and every label already on disk refers to them by number.

LAYOUTS. Both of the shapes these folders come in are read and written in place:

    images/ + labels/   <folder>/images/*.jpg  with  <folder>/labels/*.txt
    flat                <folder>/*.jpg         with  <folder>/*.txt beside them

Optional annotated/, crops/ and REVIEW.csv are kept in step when they already exist, and are
never created.

STATE. A frame is in exactly one of these, which is what the gallery tabs filter on:

    unassigned   no label file at all - nobody has looked at this frame yet
    annotated    a label file with at least one box
    null         a label file that is empty - verified as having nothing in it
    rejected     listed in .labeller/rejected.txt - unusable frame, excluded from export

Usage:
  ./server.py                                  browse from your home directory
  ./server.py --root /path/to/folders          open with that folder selected
  ./server.py --base /mnt/data --port 8001     browse anywhere under /mnt/data
  ./server.py --root ... --model weights.pt    enable assisted candidates
"""

import argparse
import csv
import hashlib
import json
import os
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

BASE: Path = Path.home()          # the sandbox the browser may not climb out of
ROOT: Path | None = None          # the folder currently being worked in
MODEL_PATH: str | None = None
_model = None
_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

IMG_EXT = (".jpg", ".jpeg", ".png")
FIELDS = ["image", "labelled", "score", "box_w_px", "found_by", "note"]
SIDE = ".labeller"                # per-folder state the tool owns; never training data
THUMB_CACHE = Path.home() / ".cache" / "drone-labeler" / "thumbs"

DEFAULT_CLASSES = ["object"]      # only used when the folder names none of its own
PALETTE = ["#efa83e", "#52c08a", "#5aa9e6", "#b87ddb", "#e0776a", "#d9c04a", "#7fd1c4",
           "#e07ab0", "#8fa6c4", "#c4a77f"]


# ---------------------------------------------------------------- paths

def inside(p: Path, parent: Path) -> bool:
    p, parent = p.resolve(), parent.resolve()
    return p == parent or parent in p.parents


def active_root() -> Path:
    if ROOT is None:
        raise HTTPException(400, "no folder selected yet")
    return ROOT


def folder_dir(folder: str) -> Path:
    """Resolve a folder id (its path relative to the active root) to a real directory."""
    root = active_root()
    d = (root / folder).resolve() if folder not in ("", ".") else root.resolve()
    if not inside(d, root):
        raise HTTPException(400, "folder outside the selected root")
    if not d.is_dir():
        raise HTTPException(404, f"no such folder: {folder}")
    return d


def image_dir(d: Path) -> Path:
    """Where this folder's images live: images/ when it has one, else the folder itself."""
    return d / "images" if (d / "images").is_dir() else d


def label_dir(d: Path, make: bool = False) -> Path:
    """
    Where this folder's labels live.

    The images/ + labels/ layout keeps them in a sibling directory. A flat folder keeps the
    .txt beside the .jpg, which is the other convention YOLO tools use and the one that does
    not surprise anyone looking in the folder afterwards.
    """
    if (d / "images").is_dir():
        ld = d / "labels"
        if make:
            ld.mkdir(exist_ok=True)
        return ld
    return d


def images_of(d: Path):
    return sorted((p for p in image_dir(d).iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXT),
                  key=lambda q: (frame_no(q.name), q.name))


def frame_no(name: str) -> int:
    """
    The trailing run of digits only.

    Joining every digit in the name gives 400001 for "fixed outdoor vid-4 dji mavic_00001" -
    the 4 from "vid-4" leaks in and the ordering silently changes.
    """
    m = re.findall(r"(\d+)", Path(name).stem)
    return int(m[-1]) if m else 0


def side_file(d: Path, name: str, make: bool = False) -> Path:
    s = d / SIDE
    if make:
        s.mkdir(exist_ok=True)
    return s / name


def rejected_set(d: Path) -> set:
    f = side_file(d, "rejected.txt")
    if not f.exists():
        return set()
    return {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}


def write_rejected(d: Path, names: set):
    f = side_file(d, "rejected.txt", make=True)
    f.write_text("".join(sorted(n + "\n" for n in names)))


# ---------------------------------------------------------------- classes

def classes_file() -> Path:
    return active_root() / "classes.txt"


def classes_from_yaml(p: Path):
    """
    `names:` out of a dataset yaml, in either shape Ultralytics accepts.

    Written by hand rather than with a yaml parser so the tool has no import to satisfy for
    one field: names is either a mapping of index to name, or a plain list.
    """
    try:
        text = p.read_text()
    except OSError:
        return None
    # [^\S\n]* and not \s*: \s matches the newline, so a greedy \s* runs past "names:" and
    # captures the first line of the block beneath it as though it were written inline.
    m = re.search(r"^names:[^\S\n]*(.*)$", text, re.M)
    if not m:
        return None
    inline = m.group(1).strip()
    if inline.startswith("[") or inline.startswith("{"):
        body = inline[1:-1] if inline[-1:] in "]}" else inline[1:]
        out = []
        for part in body.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(re.sub(r"^\d+\s*:\s*", "", part).strip("'\""))
        return out or None
    out = []
    for ln in text[m.end():].splitlines():
        if not ln.strip():
            continue
        if not ln.startswith((" ", "\t", "-")):
            break
        s = ln.strip().lstrip("-").strip()
        s = re.sub(r"^\d+\s*:\s*", "", s)
        if s:
            out.append(s.strip("'\""))
    return out or None


def discover_classes(root: Path):
    """
    What this dataset already calls its classes, from whatever file it keeps them in.

    Searched upwards as well as in place, because a YOLO dataset keeps data.yaml at its own
    root while the folder being labelled is usually a split inside it - train/, or one clip's
    folder several levels down. Only classes.txt in the selected folder wins outright, since
    that is the file this tool writes.
    """
    f = root / "classes.txt"
    if f.exists():
        names = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
        if names:
            return names, "classes.txt"
    here = root
    for up in range(4):
        if not inside(here, BASE):
            break
        for y in sorted(here.glob("*.yaml")) + sorted(here.glob("*.yml")):
            names = classes_from_yaml(y)
            if names:
                rel = y.name if up == 0 else f"{'../' * up}{y.name}"
                return names, rel
        nj = here / "notes.json"                   # what COCO-style exports leave behind
        if nj.exists():
            try:
                data = json.loads(nj.read_text())
                names = [c["name"] for c in data.get("categories", []) if c.get("name")]
                if names:
                    return names, ("notes.json" if up == 0 else f"{'../' * up}notes.json")
            except (ValueError, KeyError, TypeError):
                pass
        if here.parent == here:
            break
        here = here.parent
    return list(DEFAULT_CLASSES), None


def read_classes() -> list:
    return discover_classes(active_root())[0]


def classes_payload() -> dict:
    names, src = discover_classes(active_root())
    return {"classes": [{"id": i, "name": n, "color": PALETTE[i % len(PALETTE)]}
                        for i, n in enumerate(names)],
            "source": src}


# ---------------------------------------------------------------- labels

def read_boxes(lp: Path, w: int, h: int):
    if not lp.exists():
        return []
    out = []
    for line in lp.read_text().splitlines():
        t = line.split()
        if len(t) != 5:
            continue
        try:
            cls = int(float(t[0]))
            cx, cy, bw, bh = (float(v) for v in t[1:])
        except ValueError:
            continue
        out.append({"x0": round((cx - bw / 2) * w, 1), "y0": round((cy - bh / 2) * h, 1),
                    "x1": round((cx + bw / 2) * w, 1), "y1": round((cy + bh / 2) * h, 1),
                    "cls": cls})
    return out


def write_boxes(lp: Path, boxes, w: int, h: int) -> int:
    lines = []
    for b in boxes:
        x0, x1 = sorted((max(0.0, float(b["x0"])), min(float(w), float(b["x1"]))))
        y0, y1 = sorted((max(0.0, float(b["y0"])), min(float(h), float(b["y1"]))))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        lines.append("%d %.6f %.6f %.6f %.6f"
                     % (max(0, int(b.get("cls", 0))),
                        (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h))
    lp.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def label_counts(d: Path, stem: str):
    """(has_label, n_boxes, class ids present) for one frame, without loading the image."""
    lp = label_dir(d) / (stem + ".txt")
    if not lp.exists():
        return False, 0, []
    n, cls = 0, set()
    for line in lp.read_text().splitlines():
        t = line.split()
        if len(t) != 5:
            continue
        n += 1
        try:
            cls.add(int(float(t[0])))
        except ValueError:
            pass
    return True, n, sorted(cls)


def status_of(has_label, n_boxes, name, rejected):
    if name in rejected:
        return "rejected"
    if not has_label:
        return "unassigned"
    return "annotated" if n_boxes else "null"


def redraw(d: Path, name: str, boxes):
    """Keep annotated/ and crops/ in step, but only if the folder already has them."""
    if not (d / "annotated").is_dir() and not (d / "crops").is_dir():
        return
    im = cv2.imread(str(image_dir(d) / name))
    if im is None:
        return
    h, w = im.shape[:2]
    for b in boxes:
        cv2.rectangle(im, (int(b["x0"]) - 2, int(b["y0"]) - 2),
                      (int(b["x1"]) + 2, int(b["y1"]) + 2), (0, 0, 255), 1)
    if (d / "annotated").is_dir():
        cv2.imwrite(str(d / "annotated" / name), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if (d / "crops").is_dir():
        C = 640
        if boxes:
            cx = sum((b["x0"] + b["x1"]) / 2 for b in boxes) / len(boxes)
            cy = sum((b["y0"] + b["y1"]) / 2 for b in boxes) / len(boxes)
        else:
            cx, cy = w / 2, h / 2
        X0 = int(max(0, min(cx - C / 2, w - C)))
        Y0 = int(max(0, min(cy - C / 2, h - C)))
        crop = np.full((C, C, 3), 114, np.uint8)
        piece = im[Y0:Y0 + C, X0:X0 + C]
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


# ---------------------------------------------------------------- browsing

@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "static" / "index.html").read_text()


@app.get("/api/state")
def state():
    return {"base": str(BASE), "root": str(ROOT) if ROOT else None,
            "model": bool(MODEL_PATH)}


def scan(d: Path) -> dict:
    """Does this directory hold images, and does it already have labels beside them?"""
    try:
        imgs = [p for p in image_dir(d).iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXT]
    except (PermissionError, OSError):
        return {"images": 0, "labelled": 0, "layout": None}
    if not imgs:
        return {"images": 0, "labelled": 0, "layout": None}
    ld = label_dir(d)
    labelled = sum(1 for p in imgs if (ld / (p.stem + ".txt")).exists())
    return {"images": len(imgs), "labelled": labelled,
            "layout": "images/ + labels/" if (d / "images").is_dir() else "flat"}


@app.get("/api/browse")
def browse(path: str = ""):
    """Directories under `path`, each reporting whether it is worth opening."""
    d = Path(path).expanduser().resolve() if path else BASE.resolve()
    if not inside(d, BASE):
        raise HTTPException(400, "outside the browsable base")
    if not d.is_dir():
        raise HTTPException(404, "no such directory")
    kids = []
    try:
        for p in sorted(d.iterdir(), key=lambda q: q.name.lower()):
            if not p.is_dir() or p.name.startswith(".") or p.name == SIDE:
                continue
            info = scan(p)
            # a folder with no images of its own may still be a parent of folders that have
            deep = 0
            if not info["images"]:
                try:
                    deep = sum(1 for q in p.rglob("*")
                               if q.is_file() and q.suffix.lower() in IMG_EXT)
                except (PermissionError, OSError):
                    deep = 0
            kids.append({"name": p.name, "path": str(p), **info, "deep_images": deep})
    except PermissionError:
        raise HTTPException(403, "not readable")
    return {"path": str(d), "parent": str(d.parent) if inside(d.parent, BASE) else None,
            "self": scan(d), "dirs": kids}


class RootReq(BaseModel):
    path: str


@app.post("/api/root")
def set_root(req: RootReq):
    global ROOT
    d = Path(req.path).expanduser().resolve()
    if not inside(d, BASE):
        raise HTTPException(400, "outside the browsable base")
    if not d.is_dir():
        raise HTTPException(404, "no such directory")
    ROOT = d
    return {"root": str(ROOT), "folders": folders()}


@app.get("/api/folders")
def folders():
    """
    Every folder of images under the selected root, at any depth, plus the root itself.

    Frame folders arrive grouped in all sorts of ways - some sit directly under the root,
    others a level down inside "Mileisght frames on 2592x1944". A folder is named by its path
    relative to the root, which is what every other endpoint accepts.
    """
    root = active_root()
    seen, out = set(), []

    def add(d: Path):
        info = scan(d)
        if not info["images"] or d in seen:
            return
        seen.add(d)
        rel = "." if d == root else d.relative_to(root).as_posix()
        rj = rejected_set(d)
        counts = {"unassigned": 0, "annotated": 0, "null": 0, "rejected": 0}
        for p in images_of(d):
            has, n, _ = label_counts(d, p.stem)
            counts[status_of(has, n, p.name, rj)] += 1
        out.append({"name": rel, "count": info["images"], "layout": info["layout"], **counts})

    # images/ is half of its parent's layout, not a folder in its own right. Listing it
    # separately showed every dataset folder twice - once correctly with its labels, and once
    # as a flat folder where all 1,052 frames looked unassigned.
    skip = {"images", "labels", "annotated", "crops", SIDE}
    add(root)
    for p in sorted(root.rglob("*")):
        if p.is_dir() and not p.name.startswith(".") and p.name not in skip \
                and not any(q in skip for q in p.relative_to(root).parts):
            add(p)
    return out


# ---------------------------------------------------------------- frames

@app.get("/api/classes")
def get_classes():
    return classes_payload()


class ClassReq(BaseModel):
    classes: list


@app.post("/api/classes")
def set_classes(req: ClassReq):
    names, seen = [], set()
    for raw in req.classes:
        n = str(raw).strip().replace("\n", " ")
        if not n or n.lower() in seen:
            continue
        seen.add(n.lower())
        names.append(n)
    if not names:
        raise HTTPException(400, "need at least one class")

    # Class indices are positional, and labels on disk refer to a class by its number. So the
    # test has to be positional too: changing the NAME at an index is a rename and is safe,
    # because every box pointing there still means the same thing. Moving a name to a different
    # index, or shortening the list, is not - those boxes now point at something else, or at
    # nothing. Comparing by name instead reports a rename as a deletion, which is noise.
    before = read_classes()
    moved = [n for n in before if n in names and names.index(n) != before.index(n)]
    lost = max(0, len(before) - len(names))

    classes_file().write_text("".join(n + "\n" for n in names))
    out = classes_payload()
    out["warning"] = ""
    bits = []
    if moved:
        bits.append(f"{len(moved)} class(es) changed index ({', '.join(moved[:4])})")
    if lost:
        bits.append(f"the list is {lost} shorter, so boxes with class {len(names)} and above "
                    f"no longer point at anything")
    if bits:
        out["warning"] = ("; ".join(bits) + " - labels already on disk refer to classes by "
                          "number, so those boxes now mean something else")
    return out


@app.get("/api/frames")
def frames(folder: str, status: str = "all", page: int = 1, per: int = 60, q: str = ""):
    d = folder_dir(folder)
    rj = rejected_set(d)
    notes = {}
    f = d / "REVIEW.csv"
    if f.exists():
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            notes[r.get("image", "")] = r.get("note", "")
    items, counts = [], {"all": 0, "unassigned": 0, "annotated": 0, "null": 0, "rejected": 0}
    needle = q.strip().lower()
    for p in images_of(d):
        has, n, cls = label_counts(d, p.stem)
        st = status_of(has, n, p.name, rj)
        counts[st] += 1
        counts["all"] += 1
        if needle and needle not in p.name.lower():
            continue
        if status not in ("all", st):
            continue
        items.append({"name": p.name, "n": frame_no(p.name), "boxes": n, "status": st,
                      "classes": cls, "note": notes.get(p.name, "")})
    per = max(1, min(400, per))
    pages = max(1, (len(items) + per - 1) // per)
    page = max(1, min(page, pages))
    return {"total": len(items), "page": page, "pages": pages, "per": per,
            "counts": counts, "items": items[(page - 1) * per: page * per]}


def hex_bgr(h: str):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


@app.get("/api/thumb")
def thumb(folder: str, name: str, w: int = 320, ann: int = 1):
    """
    A card-sized copy, with the boxes drawn on by default.

    Drawing them matters: a wall of bare thumbnails tells you an image is annotated but not
    whether the box is in the right place, which is the thing worth seeing at a glance.
    """
    d = folder_dir(folder)
    p = image_dir(d) / name
    if not p.exists():
        raise HTTPException(404, "no such frame")
    w = max(80, min(640, w))
    lp = label_dir(d) / (Path(name).stem + ".txt")
    lstamp = lp.stat().st_mtime_ns if (ann and lp.exists()) else 0
    key = hashlib.sha1(f"{p}|{p.stat().st_mtime_ns}|{w}|{ann}|{lstamp}".encode()).hexdigest()
    cached = THUMB_CACHE / key[:2] / (key + ".jpg")
    if cached.exists():
        return FileResponse(cached, media_type="image/jpeg")
    im = cv2.imread(str(p))
    if im is None:
        raise HTTPException(415, "unreadable image")
    h0, w0 = im.shape[:2]
    scale = w / w0
    small = cv2.resize(im, (w, max(1, int(round(h0 * scale)))), interpolation=cv2.INTER_AREA)
    if ann:
        for b in read_boxes(lp, w0, h0):
            col = hex_bgr(PALETTE[b["cls"] % len(PALETTE)])
            x0, y0 = int(b["x0"] * scale), int(b["y0"] * scale)
            x1, y1 = int(b["x1"] * scale), int(b["y1"] * scale)
            # a 6 px drone shrinks to nothing on a card, so tiny boxes get a reticle instead
            if x1 - x0 < 6 and y1 - y0 < 6:
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                cv2.rectangle(small, (cx - 7, cy - 7), (cx + 7, cy + 7), col, 1)
            else:
                cv2.rectangle(small, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), col, 2)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(buf.tobytes())
    except OSError:
        pass
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.get("/api/image")
def image(folder: str, name: str, enhance: int = 0, clip: float = 4.0):
    d = folder_dir(folder)
    p = image_dir(d) / name
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
    im = cv2.imread(str(image_dir(d) / name))
    if im is None:
        raise HTTPException(404, "no such frame")
    h, w = im.shape[:2]
    lp = label_dir(d) / (Path(name).stem + ".txt")
    return {"w": int(w), "h": int(h), "boxes": read_boxes(lp, w, h),
            "has_label": lp.exists(), "rejected": name in rejected_set(d)}


class SaveReq(BaseModel):
    folder: str
    name: str
    boxes: list
    note: str = ""


@app.post("/api/label")
def save_label(req: SaveReq):
    d = folder_dir(req.folder)
    im = cv2.imread(str(image_dir(d) / req.name))
    if im is None:
        raise HTTPException(404, "no such frame")
    h, w = im.shape[:2]
    lp = label_dir(d, make=True) / (Path(req.name).stem + ".txt")
    n = write_boxes(lp, req.boxes, w, h)
    widest = max((b["x1"] - b["x0"] for b in req.boxes), default=0)
    redraw(d, req.name, req.boxes)
    update_review(d, req.name, n, widest,
                  req.note or ("" if n else "nothing in frame - marked by hand"))
    return {"ok": True, "boxes": n}


class StatusReq(BaseModel):
    folder: str
    names: list
    status: str                 # "rejected" to reject, "unassigned" to undo, "null" to mark


@app.post("/api/status")
def set_status(req: StatusReq):
    d = folder_dir(req.folder)
    rj = rejected_set(d)
    changed = 0
    for name in req.names:
        if not (image_dir(d) / name).exists():
            continue
        if req.status == "rejected":
            rj.add(name)
        else:
            rj.discard(name)
            if req.status == "null":
                lp = label_dir(d, make=True) / (Path(name).stem + ".txt")
                lp.write_text("")
                redraw(d, name, [])
                update_review(d, name, 0, 0, "nothing in frame - marked by hand")
            elif req.status == "unassigned":
                lp = label_dir(d) / (Path(name).stem + ".txt")
                if lp.exists():
                    lp.unlink()
        changed += 1
    write_rejected(d, rj)
    return {"ok": True, "changed": changed}


class DelReq(BaseModel):
    folder: str
    names: list


@app.post("/api/delete")
def delete_frames(req: DelReq):
    d = folder_dir(req.folder)
    ld = label_dir(d)
    removed = 0
    for name in req.names:
        stem = Path(name).stem
        for p in (image_dir(d) / name, ld / (stem + ".txt"),
                  d / "annotated" / name, d / "crops" / name):
            if p.exists() and p.is_file():
                p.unlink()
        removed += 1
    f = d / "REVIEW.csv"
    if f.exists():
        rows = [r for r in csv.DictReader(open(f, encoding="utf-8-sig"))
                if r.get("image") not in set(req.names)]
        with open(f, "w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows([{k: r.get(k, "") for k in FIELDS} for r in rows])
    rj = rejected_set(d) - set(req.names)
    write_rejected(d, rj)
    return {"ok": True, "removed": removed}


# ---------------------------------------------------------------- assistance

class AiReq(BaseModel):
    folder: str
    name: str


@app.post("/api/ai")
def ai(req: AiReq):
    """Candidates from three independent sources. Suggestions only - nothing is written."""
    global _model
    d = folder_dir(req.folder)
    ps = images_of(d)
    names = [p.name for p in ps]
    if req.name not in names:
        raise HTTPException(404, "no such frame")
    i = names.index(req.name)
    g = cv2.imread(str(ps[i]), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise HTTPException(415, "unreadable image")
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
        a = cv2.imread(str(ps[i - 1]), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(ps[i + 1]), cv2.IMREAD_GRAYSCALE)
        if a is not None and b is not None and a.shape == g.shape == b.shape:
            dd = np.minimum(a.astype(np.int16) - g, b.astype(np.int16) - g)
            dd = cv2.GaussianBlur(np.clip(dd, 0, 255).astype(np.uint8), (3, 3), 0)
            thr = max(6, int(np.percentile(dd, 99.9)))
            nn, _, st, _ = cv2.connectedComponentsWithStats((dd > thr).astype(np.uint8), 8)
            for k in range(1, nn):
                x, y, w, h, area = (int(v) for v in st[k])
                if 3 <= max(w, h) <= 260 and area >= 4:
                    out.append({"box": [x, y, x + w, y + h], "src": "motion",
                                "score": round(float(dd[y:y + h, x:x + w].max()) / 255, 3)})

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
                            "score": round(float(z[y:y + h, x:x + w].max()) / 20, 3)})

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
    g = cv2.imread(str(image_dir(d) / req.name), cv2.IMREAD_GRAYSCALE)
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
                dist = float(np.hypot(x0 + x + w / 2 - sx, y0 + y + h / 2 - sy))
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
    global BASE, ROOT, MODEL_PATH
    ap = argparse.ArgumentParser(description="Local web labeller for folders of frames.")
    ap.add_argument("--root", type=Path, default=None,
                    help="folder to open with (otherwise pick one in the browser)")
    ap.add_argument("--base", type=Path, default=None,
                    help="the only tree the folder browser may enter (default: your home)")
    ap.add_argument("--model", default=None,
                    help="optional YOLO weights, enables assisted candidates")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.root is not None and not args.root.is_dir():
        sys.exit(f"no such directory: {args.root}")
    BASE = (args.base or (args.root.resolve().parent if args.root else Path.home())).resolve()
    if not BASE.is_dir():
        sys.exit(f"no such directory: {BASE}")
    if args.root is not None:
        ROOT = args.root.resolve()
        if not inside(ROOT, BASE):
            sys.exit(f"--root must sit inside --base ({BASE})")
    MODEL_PATH = args.model

    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    import uvicorn
    print(f"base   {BASE}")
    print(f"root   {ROOT or 'none yet - pick a folder in the browser'}")
    print(f"model  {MODEL_PATH or 'none - assisted candidates limited to imaging'}")
    print(f"open   http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
