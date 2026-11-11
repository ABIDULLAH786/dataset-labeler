#!/usr/bin/env python3
"""
Dataset Labeler - a local web tool for drawing boxes on folders of images.

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
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
app = FastAPI(title="Dataset Labeler")


# The page and its script must always be revalidated. Without a Cache-Control header the
# browser cached app.js heuristically for hours, so after an update it ran the old script
# under the new index.html. no-cache still lets the ETag answer 304, so reloads stay cheap.
@app.middleware("http")
async def revalidate_ui(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

BASE: Path = Path.home()          # the sandbox the browser may not climb out of
ROOT: Path | None = None          # the folder currently being worked in
MODEL_PATH: str | None = None
LOCK_BASE = False            # refuse to follow a chosen folder out of --base
_model = None
_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

IMG_EXT = (".jpg", ".jpeg", ".png")
FIELDS = ["image", "labelled", "score", "box_w_px", "found_by", "note"]
SIDE = ".labeller"                # per-folder state the tool owns; never training data
THUMB_CACHE = Path.home() / ".cache" / "dataset-labeler" / "thumbs"

# No default class. A made-up "object" gets saved into real label files by whoever does not
# notice it, and a class list is the one thing the tool must not invent on someone's behalf.
# What it does instead is REMEMBER: the last list used is kept here and offered to the next
# folder that names none of its own, so a list is created once and carries across folders.
REMEMBERED = Path.home() / ".config" / "dataset-labeler" / "classes.txt"
PALETTE = ["#f5c518", "#2fb673", "#3c90e0", "#9b5fd0", "#e0614f", "#17a8a0", "#e0699f",
           "#6b7fa8", "#c08a3e", "#7cb342"]


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


# Scanning a folder means stat-ing every image and reading every label file. On 2,000 frames
# that is 200 ms, and it was being paid again on every page change, tab switch and save. The
# answer is cached against the modification times of the two directories that can invalidate
# it, so an edit is still seen immediately but an unchanged folder is free.
_SCAN: dict = {}
_WALK: dict = {}
WALK_TTL = 20.0                  # seconds a directory listing is reused for


def _stamp(d: Path) -> tuple:
    """
    A fingerprint that changes whenever anything the scan depends on changes.

    A directory's own mtime is not enough: writing into a label file that already exists does
    not touch the directory, so a cache keyed on that alone never notices an edit - it reports
    stale counts, which is worse than being slow. The newest mtime among the label files is
    the part that actually moves, and scandir gets it without opening anything.
    """
    out = []
    for sub in (image_dir(d), d / SIDE):
        try:
            out.append(sub.stat().st_mtime_ns)
        except OSError:
            out.append(0)
    newest = count = 0
    try:
        with os.scandir(label_dir(d)) as it:
            for e in it:
                if e.name.endswith(".txt"):
                    count += 1
                    m = e.stat().st_mtime_ns
                    if m > newest:
                        newest = m
    except OSError:
        pass
    return tuple(out) + (newest, count)


def scan_frames(d: Path):
    """[(name, has_label, n_boxes, classes, status)] for a folder, cached on directory mtime."""
    key = str(d)
    stamp = _stamp(d)
    hit = _SCAN.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    rj = rejected_set(d)
    rows = []
    for p in images_of(d):
        has, n, cls = label_counts(d, p.stem)
        rows.append((p.name, has, n, cls, status_of(has, n, p.name, rj)))
    _SCAN[key] = (stamp, rows)
    return rows


def forget_scan(d: Path):
    _SCAN.pop(str(d), None)


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
    if REMEMBERED.exists():
        names = [ln.strip() for ln in REMEMBERED.read_text().splitlines() if ln.strip()]
        if names:
            return names, "remembered"
    return [], None


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


def clamp(v, hi):
    return min(max(0.0, float(v)), float(hi))


def write_boxes(lp: Path, boxes, w: int, h: int) -> int:
    lines = []
    for b in boxes:
        # Clamp BOTH ends into the frame before ordering them. Clamping only the low end of
        # one and the high end of the other leaves a box that sits entirely past the edge
        # intact: max(0,1400)=1400 and min(1280,1500)=1280 sort to (1280,1400), and the file
        # then gets a centre of 1.05, outside the 0..1 a YOLO label is allowed to use.
        x0, x1 = sorted((clamp(b["x0"], w), clamp(b["x1"], w)))
        y0, y1 = sorted((clamp(b["y0"], h), clamp(b["y1"], h)))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        lines.append("%d %.6f %.6f %.6f %.6f"
                     % (max(0, int(b.get("cls", 0))),
                        (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h))
    lp.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def label_counts(d: Path, stem: str):
    """
    (has_label, n_boxes, {class: boxes of it}) for one frame, without loading the image.

    Per class rather than a set of ids: a frame with three gulls in it counts as three towards
    the bird total, and the class manager's "delete this class removes N boxes" has to be the
    number it will actually remove.
    """
    lp = label_dir(d) / (stem + ".txt")
    if not lp.exists():
        return False, 0, {}
    n, cls = 0, {}
    for line in lp.read_text().splitlines():
        t = line.split()
        if len(t) != 5:
            continue
        n += 1
        try:
            c = int(float(t[0]))
        except ValueError:
            continue
        cls[c] = cls.get(c, 0) + 1
    return True, n, cls


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
            "model": bool(MODEL_PATH), "locked": LOCK_BASE}


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
    # images/, labels/ and the rest are parts of THIS folder's layout, not folders you would
    # open on their own. Listed plainly they read as candidates, and images/ even reports
    # "0 labelled" because its labels are in the sibling directory, not inside it.
    parts = {"images": "the images of this folder", "labels": "the labels of this folder",
             "annotated": "review copies with boxes drawn on",
             "crops": "review crops"} if (d / "images").is_dir() else {}
    kids = []
    try:
        for p in sorted(d.iterdir(), key=lambda q: q.name.lower()):
            if not p.is_dir() or p.name.startswith(".") or p.name == SIDE:
                continue
            if p.name in parts:
                kids.append({"name": p.name, "path": str(p), "images": 0, "labelled": 0,
                             "layout": None, "deep_images": 0, "role": parts[p.name]})
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


@app.post("/api/pick")
def pick_folder():
    """
    Open the desktop's own folder chooser and return what was picked.

    A browser cannot hand back a real filesystem path - <input webkitdirectory> gives file
    contents and no location, by design. But this server runs on the same machine as the
    browser, so it can open the native dialog itself. Tried in order of what desktops
    actually have: zenity (GNOME), kdialog (KDE), then tkinter.
    """
    import shutil as sh
    import subprocess
    start = str(ROOT or BASE)

    for argv in ([sh.which("zenity"), "--file-selection", "--directory",
                  "--title=Choose a folder to label", f"--filename={start}/"],
                 [sh.which("kdialog"), "--getexistingdirectory", start]):
        if not argv[0]:
            continue
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0 and r.stdout.strip():
            return {"path": r.stdout.strip()}
        if r.returncode in (1, 252):              # the person cancelled
            return {"path": None}

    try:
        import tkinter
        from tkinter import filedialog
        win = tkinter.Tk()
        win.withdraw()
        win.attributes("-topmost", True)
        got = filedialog.askdirectory(initialdir=start, title="Choose a folder to label")
        win.destroy()
        return {"path": got or None}
    except Exception:
        raise HTTPException(501, "no desktop folder dialog available - "
                                 "install zenity, or type the path instead")


class RootReq(BaseModel):
    path: str


@app.post("/api/base")
def set_base(req: RootReq):
    """
    Move the browsable base, without scanning anything.

    /api/browse stays bounded so a stray URL cannot walk the disk, but a folder chosen from the
    desktop dialog or typed in deliberately is allowed to land anywhere. This is how it gets
    there: widen first, then browse normally.
    """
    global BASE
    d = Path(req.path).expanduser().resolve()
    if not d.is_dir():
        raise HTTPException(404, f"no such directory: {d}")
    if not inside(d, BASE):
        if LOCK_BASE:
            raise HTTPException(400, f"outside the locked base ({BASE})")
        BASE = d.parent if d.parent != d else d
    return {"base": str(BASE)}


@app.post("/api/root")
def set_root(req: RootReq):
    """
    Select the folder to work in.

    Picking from the desktop dialog, or typing a path, is a deliberate act by whoever is at the
    keyboard, so it is allowed to land outside the current base and the base moves to its
    parent. Only --lock-base refuses, for anyone who wants the tool pinned to one tree. Casual
    browsing is still bounded: /api/browse never leaves the base.
    """
    global ROOT, BASE
    d = Path(req.path).expanduser().resolve()
    if not d.is_dir():
        raise HTTPException(404, f"no such directory: {d}")
    if not inside(d, BASE):
        if LOCK_BASE:
            raise HTTPException(400, f"outside the locked base ({BASE})")
        BASE = d.parent if d.parent != d else d
    ROOT = d
    return {"root": str(ROOT), "base": str(BASE), "folders": folders()}


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
        counts = {"unassigned": 0, "annotated": 0, "null": 0, "rejected": 0}
        for _, _, _, _, st in scan_frames(d):
            counts[st] += 1
        out.append({"name": rel, "count": info["images"], "layout": info["layout"], **counts})

    # images/ is half of its parent's layout, not a folder in its own right. Listing it
    # separately showed every dataset folder twice - once correctly with its labels, and once
    # as a flat folder where all 1,052 frames looked unassigned.
    skip = {"images", "labels", "annotated", "crops", SIDE}
    # The walk itself is the slow half on a big tree, and folders appear and vanish far more
    # rarely than labels change. The per-folder COUNTS still come from scan_frames, which is
    # keyed on directory mtime, so an edit shows up at once - only the shape of the tree is
    # held for a few seconds.
    hit = _WALK.get(str(root))
    if hit and time.monotonic() - hit[0] < WALK_TTL:
        dirs = hit[1]
    else:
        dirs = [p for p in sorted(root.rglob("*"))
                if p.is_dir() and not p.name.startswith(".") and p.name not in skip
                and not any(q in skip for q in p.relative_to(root).parts)]
        _WALK[str(root)] = (time.monotonic(), dirs)
    add(root)
    for p in dirs:
        add(p)
    return out


# ---------------------------------------------------------------- frames

@app.get("/api/classes")
def get_classes():
    return classes_payload()


def all_label_files():
    """Every label file under the selected root, both layouts."""
    root = active_root()
    out, seen = [], set()
    for f in folders():
        d = folder_dir(f["name"])
        ld = label_dir(d)
        for p in images_of(d):
            lp = ld / (p.stem + ".txt")
            if lp.exists() and lp not in seen:
                seen.add(lp)
                out.append(lp)
    return out


@app.get("/api/class-usage")
def class_usage():
    """How many boxes point at each class index, across the whole selected root."""
    use, files = {}, 0
    for f in folders():
        d = folder_dir(f["name"])
        for _name, has, _n, cls, _st in scan_frames(d):
            if not has:
                continue
            files += 1
            for c, k in cls.items():
                use[c] = use.get(c, 0) + k
    return {"usage": use, "files": files}


class ClassItem(BaseModel):
    id: int | None = None          # the index this class had before, or null when it is new
    name: str


class ClassReq(BaseModel):
    classes: list[ClassItem]
    apply_to_labels: bool = False


@app.post("/api/classes")
def set_classes(req: ClassReq):
    """
    Save the class list, and - when asked - rewrite the labels to match.

    Classes are positional: a label file says `2 0.51 0.44 ...`, not `plane`. So renaming,
    reordering and deleting are not edits to a list of words, they change what every box
    already on disk means. Each item carries the index it HELD BEFORE, which is what makes the
    difference knowable: same id at a new position is a move, a new name on an old id is a
    rename, a missing id is a deletion. Without that the two are indistinguishable, and a
    rename gets treated as delete-plus-add.
    """
    names, ids, seen = [], [], set()
    for item in req.classes:
        n = item.name.strip().replace("\n", " ")
        if not n or n.lower() in seen:
            continue
        seen.add(n.lower())
        names.append(n)
        ids.append(item.id)
    if not names:
        raise HTTPException(400, "need at least one class")

    before = read_classes()
    try:                                          # carry this list to the next folder
        REMEMBERED.parent.mkdir(parents=True, exist_ok=True)
        REMEMBERED.write_text("".join(n + "\n" for n in names))
    except OSError:
        pass
    remap = {}                                   # old index -> new index, absent = deleted
    for new_i, old_i in enumerate(ids):
        if old_i is not None and 0 <= old_i < len(before):
            remap[old_i] = new_i
    dropped = [i for i in range(len(before)) if i not in remap]
    moved = [before[o] for o, n in remap.items() if o != n]

    changed_boxes = dropped_boxes = touched = 0
    if req.apply_to_labels and (dropped or moved):
        for lp in all_label_files():
            lines, hit = [], False
            for line in lp.read_text().splitlines():
                t = line.split()
                if len(t) != 5:
                    continue
                try:
                    c = int(float(t[0]))
                except ValueError:
                    continue
                if c in remap:
                    if remap[c] != c:
                        hit = True
                        changed_boxes += 1
                    lines.append(" ".join([str(remap[c])] + t[1:]))
                else:
                    hit = True
                    dropped_boxes += 1          # its class no longer exists: the box goes
            if hit:
                lp.write_text("\n".join(lines) + ("\n" if lines else ""))
                touched += 1

    if touched:
        _SCAN.clear()
    classes_file().write_text("".join(n + "\n" for n in names))
    out = classes_payload()
    out["warning"] = ""
    out["applied"] = {"files": touched, "moved": changed_boxes, "dropped": dropped_boxes}
    if (dropped or moved) and not req.apply_to_labels:
        bits = []
        if moved:
            bits.append(f"{len(moved)} class(es) changed index ({', '.join(moved[:4])})")
        if dropped:
            bits.append(f"{len(dropped)} deleted ({', '.join(before[i] for i in dropped[:4])})")
        out["warning"] = ("; ".join(bits) + " - labels on disk were left alone, so boxes "
                          "pointing at those indices now mean something else")
    return out


@app.get("/api/frames")
def frames(folder: str, status: str = "all", page: int = 1, per: int = 60, q: str = ""):
    d = folder_dir(folder)
    notes = {}
    f = d / "REVIEW.csv"
    if f.exists():
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            notes[r.get("image", "")] = r.get("note", "")
    items, counts = [], {"all": 0, "unassigned": 0, "annotated": 0, "null": 0, "rejected": 0}
    needle = q.strip().lower()
    for name, _has, n, cls, st in scan_frames(d):
        counts[st] += 1
        counts["all"] += 1
        if needle and needle not in name.lower():
            continue
        if status not in ("all", st):
            continue
        items.append({"name": name, "n": frame_no(name), "boxes": n, "status": st,
                      "classes": sorted(cls), "note": notes.get(name, "")})
    per = max(1, min(400, per))
    pages = max(1, (len(items) + per - 1) // per)
    page = max(1, min(page, pages))
    return {"total": len(items), "page": page, "pages": pages, "per": per,
            "counts": counts, "items": items[(page - 1) * per: page * per]}


def hex_bgr(h: str):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


@app.get("/api/thumb")
def thumb(request: Request, folder: str, name: str, w: int = 320, ann: int = 1):
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
    # The key already covers both mtimes, so the answer for a given URL+ETag can never go
    # stale: an edited label changes the key. That lets the browser keep the picture and skip
    # the request entirely, which is the difference between a page of 60 cards redrawing
    # instantly and re-fetching 60 files every time a tab is clicked.
    etag = f'"{key}"'
    headers = {"ETag": etag, "Cache-Control": "private, max-age=86400"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    cached = THUMB_CACHE / key[:2] / (key + ".jpg")
    if cached.exists():
        return FileResponse(cached, media_type="image/jpeg", headers=headers)
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
    return Response(buf.tobytes(), media_type="image/jpeg", headers=headers)


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
    forget_scan(d)
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
    forget_scan(d)
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
    forget_scan(d)
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
    global BASE, ROOT, MODEL_PATH, LOCK_BASE
    ap = argparse.ArgumentParser(description="Local web labeller for folders of frames.")
    ap.add_argument("--root", type=Path, default=None,
                    help="folder to open with (otherwise pick one in the browser)")
    ap.add_argument("--base", type=Path, default=None,
                    help="where the folder browser starts (default: your home)")
    ap.add_argument("--lock-base", action="store_true",
                    help="refuse to open any folder outside --base")
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
    LOCK_BASE = args.lock_base

    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    import uvicorn
    print(f"base   {BASE}")
    print(f"root   {ROOT or 'none yet - pick a folder in the browser'}")
    print(f"model  {MODEL_PATH or 'none - assisted candidates limited to imaging'}")
    print(f"open   http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
