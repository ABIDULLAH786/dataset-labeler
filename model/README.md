# model/

Put the YOLO weights you want **AI assist** to use in this folder.

## Which file is used

When the labeller starts, it picks your model as the first of these that exists:

1. a file given with `--model path/to/weights.pt`, or the `LABELLER_MODEL` environment variable
2. `model/best.pt`
3. the first `*.pt` in this folder, sorted by name

So name your model `best.pt` if you keep more than one here. With a single `.pt`, any name works.
The header shows the one in use (`model: <name>`), or `no model in model/`.

## Choosing what AI assist runs

The picker beside **AI assist** has four options:

| Option | Runs |
|---|---|
| **All sources** (default) | your model, COCO, and the motion and contrast finds |
| **Your model** | only your weights from this folder |
| **COCO default** | only COCO: a general YOLO11 trained on the 80 COCO classes |
| **Imaging only** | no model: motion between neighbouring frames, and local contrast |

**Your model** is greyed out, with the reason, when there is no `.pt` here. Both model options are
greyed out when Ultralytics is not installed, or when the labeller was started with `--no-model`.

In the candidate list, your model's finds always come first, then COCO's, then motion and
contrast. Each model find shows its source (`model` or `coco`) and the class it found. Accepting
one gives it the dataset class of the same name if there is one (a model's `drone` becomes your
`drone`); otherwise it gets the class you are drawing in.

## COCO: downloaded on first use, not with the code

COCO is not part of the repository: it is a weights file (`yolo11m.pt`, about 40 MB), and model
weights do not belong in git. Cloning the labeller does not fetch it.

The first time AI assist runs with COCO (All sources, or COCO default), Ultralytics downloads it
once into `~/.cache/drone-labeler/weights/`. Every later run reuses that copy. It is kept out of
this folder on purpose, so a downloaded COCO file is never mistaken for your own model. The
picker says `downloads ~40 MB once` until that has happened.

To download it ahead of time, for example before working offline:

```bash
mkdir -p ~/.cache/drone-labeler/weights && cd ~/.cache/drone-labeler/weights && \
  /path/to/drone-labeler/.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
```

To use another COCO size, set `LABELLER_COCO`, for example `LABELLER_COCO=yolo11x.pt` (slower,
finds more small objects) or `LABELLER_COCO=yolo11n.pt` (fastest).

COCO has no "drone" class. It is useful for birds, aircraft and kites, and to see how AI assist
works before you have a model of your own.

## Requirements

The model options need Ultralytics, which is optional:

```bash
.venv/bin/pip install -r requirements-ai.txt
```

`.pt` files in this folder are ignored by git.
