# Drone Labeller

A local web tool for boxing drones in folders of video frames, built for targets that are only
a handful of pixels across.

Pointing at an 8 px drone in words does not work. This serves the frames in a browser, lets the
box be drawn where the drone actually is, and writes the YOLO label straight back to disk beside
the image.

![interface](docs/screenshot.png)

## Why it is not a generic annotation tool

Three things here exist because of what these frames are.

**Contrast on demand.** Many of the hard frames have the drone in shade against a building,
where the region averages 52 out of 255 and nothing is visible in the raw image. Pressing `E`
serves the frame through CLAHE and the drone appears. The enhancement is **for looking only** —
the image on disk is never modified.

**Snap.** Clicking near the drone turns the click into a box that hugs whatever is actually
there. Labels stay tight instead of carrying a lot of empty space around the target, which
matters when the target is 10 px wide and the box is 40.

**Assisted candidates, not an oracle.** `A` runs a detection model, frame differencing and a
local z-score, and lists what each found. They are suggestions to accept or ignore. The model is
never trusted on its own — it is usually the model being retrained, which is why the frames need
labelling in the first place.

## Install

```bash
pip install -r requirements.txt
# optional, for model-assisted candidates:
pip install ultralytics
```

## Run

```bash
python3 server.py --root /path/to/folders
python3 server.py --root /path/to/folders --model /path/to/weights.pt --port 8001
```

Then open `http://127.0.0.1:8000`.

## Folder layout

The tool reads and writes this layout and leaves everything else alone. Only `images/` is
required; the rest are created or updated when present.

```
<root>/
  <folder>/
    images/*.jpg        the frames
    labels/*.txt        YOLO, class 0 — created when missing
    annotated/*.jpg     boxed copies, kept in step if the folder exists
    crops/*.jpg         640x640 review crops, kept in step if the folder exists
    REVIEW.csv          per-frame notes, kept in step if present
```

An **empty** label file is meaningful: it marks the frame as a verified negative. That is not
the same as having no label file at all, which means nobody has looked yet. The frame list
distinguishes the two.

## Using it

| | |
|---|---|
| **click** | snap a box onto whatever is under the cursor |
| **drag** | draw a box by hand |
| **click a box** | select it — then drag to move, or pull a handle to resize |
| **wheel** | zoom at the cursor |
| **space-drag**, middle-drag | pan |
| **loupe** | follows the cursor at 8x, so small targets are clickable |

### Keys

`→` / `D` next · `←` prev · `S` save · `B` no drone · `A` assisted candidates · `E` contrast ·
`F` fit · `Del` remove selected box · `Esc` deselect

### Two behaviours to know

**Navigating saves.** There is no unsaved-changes prompt. A confirmation on every frame is
friction in a job that runs to hundreds of frames.

**Next and previous follow the filter.** With **Blank** selected, Next goes to the next blank
frame, not the next frame in the folder. The counter in the toolbar shows the position within
the filtered set. **Check** lists frames an automated pass flagged as weak evidence, which is
usually where to start.

## Frame status

| Dot | Meaning |
|---|---|
| green | has one or more boxes |
| grey | empty label — verified as having no drone |
| red | no label file — nobody has looked at this frame yet |

## Endpoints

Everything the interface does is a plain HTTP call, so the same operations can be scripted.

| | |
|---|---|
| `GET /api/folders` | folders under the root, with frame counts |
| `GET /api/frames?folder=` | frames with box counts, label state and notes |
| `GET /api/image?folder=&name=&enhance=0\|1` | the frame, optionally contrast-enhanced |
| `GET /api/label?folder=&name=` | current boxes in pixel coordinates |
| `POST /api/label` | write boxes; an empty list writes a negative |
| `POST /api/snap` | `{x, y}` to a tight box |
| `POST /api/ai` | candidates from model, differencing and local contrast |
| `POST /api/delete` | remove a frame and everything derived from it |
