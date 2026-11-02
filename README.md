# Drone Labeller

A local web tool for drawing boxes on folders of images, built for targets that are only a
handful of pixels across.

Pointing at an 8 px drone in words does not work. This serves any folder of images in a browser,
lets the box be drawn where the target actually is, and writes the YOLO label straight back to
disk beside the image.

![interface](docs/screenshot.png)

## Why it is not a generic annotation tool

Three things here exist because of what these frames are.

**Contrast on demand.** Many hard frames have the target in shade against a building, where the
region averages 52 out of 255 and nothing is visible in the raw image. Pressing `E` serves the
frame through CLAHE and it appears. The enhancement is **for looking only** — the image on disk
is never modified.

**Snap.** Clicking near the target turns the click into a box that hugs whatever is actually
there. Labels stay tight instead of carrying a lot of empty space, which matters when the target
is 10 px wide and the box is 40.

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
python3 server.py                                   # browse from your home directory
python3 server.py --root /path/to/folder            # open with that folder selected
python3 server.py --base /mnt/data --port 8001      # browse anywhere under /mnt/data
python3 server.py --root ... --model weights.pt     # enable assisted candidates
```

Then open `http://127.0.0.1:8000`.

`--base` is the only tree the folder browser may enter, so the tool cannot be pointed at the
rest of the filesystem by a stray URL. It defaults to your home directory, or to the parent of
`--root` when you pass one.

## Choosing a folder

Any folder holding images works, and both layouts are read and written **in place**:

```
images/ + labels/     <folder>/images/*.jpg   with   <folder>/labels/*.txt
flat                  <folder>/*.jpg          with   <folder>/*.txt beside them
```

The browser shows, for every directory, how many images it holds, how many are already
labelled, and which layout it is — so you can tell a dataset folder from its parent before
opening it. Folders with no images of their own report how many sit deeper inside.

**A folder with images and no label files at all is perfectly fine.** Every frame simply starts
as **Unassigned**, nothing is written until you label something, and the `labels/` directory is
created on the first save. Nothing in the folder is touched before that.

Optional `annotated/`, `crops/` and `REVIEW.csv` are kept in step when they already exist, and
are never created.

## Classes

Classes belong to the dataset, not to the tool. On opening a folder the class list is read from
whatever that dataset already carries, searching upward as far as four levels because a YOLO
dataset keeps `data.yaml` at its root while the folder you are labelling is usually a split
inside it:

1. `classes.txt` in the selected folder — what this tool writes, and it wins outright
2. any `*.yaml` / `*.yml` with a `names:` field, in block or inline form
3. `notes.json`, which COCO-style exports leave behind

Classes are edited from the **Classes** button, one per line. Pick the active class from the
header chips, with `↑`/`↓` or with the number keys, and every new box is drawn with it. Changing
the class of a selected box is the same action — pick a class while a box is selected.

> **Class indices are positional.** A label file says `2 0.51 0.44 0.03 0.02`, not `plane`.
> Renaming a class in place is safe and adding one at the end is safe. **Reordering the list, or
> shortening it, silently changes what every box already on disk means** — the tool saves the
> change and tells you plainly what it just did.

## The gallery

Folders can run to thousands of frames, so the gallery opens first: cards in a grid, paged, with
the boxes drawn on each thumbnail. A wall of bare thumbnails tells you an image is annotated but
not whether the box is in the right place, which is the thing worth seeing at a glance. Boxes
too small to show on a card are drawn as a reticle instead.

Tabs filter by state, and each frame is in exactly one:

| Tab | Meaning |
|---|---|
| **Unassigned** | no label file at all — nobody has looked at this frame yet |
| **Annotated** | a label file with at least one box |
| **Null** | a label file that is empty — verified as having nothing in it |
| **Rejected** | listed in `.labeller/rejected.txt` — unusable frame, excluded from export |

An empty label file is **not** the same as a missing one. One says "I looked and there is
nothing here", which is a training negative; the other says "nobody has looked".

Rejecting keeps the label file. Rejection is a decision about the dataset, not about the
labelling, and the tool records it in a sidecar rather than deleting your work. Nothing in
`.labeller/` is training data.

Select cards with their checkboxes for the batch bar: mark null, reject, un-reject, clear labels
(back to Unassigned), or delete from disk. Deleting always asks first and names what goes.

## Annotating

| | |
|---|---|
| **click** | snap a box onto whatever is under the cursor, in the active class |
| **drag** | draw a box by hand |
| **click a box** | select it — then drag to move, or pull a handle to resize |
| **wheel** | zoom at the cursor |
| **space-drag**, middle-drag | pan |
| **loupe** | follows the cursor at 8×, so small targets are clickable |

### Keys

`→` / `D` next · `←` prev · `↑` `↓` cycle class · `2`…`9` pick a class · `S` save · `N` mark
null · `R` reject · `A` assisted candidates · `E` contrast · `0` fit · `1` 100% · `Del` remove
selected box · `Esc` deselect

### Two behaviours to know

**Navigating saves.** There is no unsaved-changes prompt. A confirmation on every frame is
friction in a job that runs to hundreds of frames.

**Next and previous follow the tab and the page.** With **Unassigned** selected, Next goes to
the next unassigned frame. The counter shows the position within that filtered set.

## Endpoints

Everything the interface does is a plain HTTP call, so the same operations can be scripted.

| | |
|---|---|
| `GET /api/state` | the browsable base, the selected root, whether a model is loaded |
| `GET /api/browse?path=` | directories under a path, with image and label counts |
| `POST /api/root` | select the folder to work in |
| `GET /api/folders` | image folders under the root, with per-status counts |
| `GET /api/classes` · `POST /api/classes` | read and write the class list |
| `GET /api/frames?folder=&status=&page=&per=&q=` | one page of frames, with tab counts |
| `GET /api/thumb?folder=&name=&w=&ann=` | card-sized copy, boxes drawn by default |
| `GET /api/image?folder=&name=&enhance=` | the frame, optionally contrast-enhanced |
| `GET /api/label` · `POST /api/label` | boxes in pixel coordinates, with class indices |
| `POST /api/status` | mark rejected, null, or back to unassigned, in batches |
| `POST /api/delete` | remove frames and everything derived from them, in batches |
| `POST /api/snap` | `{x, y}` to a tight box |
| `POST /api/ai` | candidates from model, differencing and local contrast |
