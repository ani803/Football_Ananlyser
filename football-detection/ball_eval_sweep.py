"""
ball_eval_sweep.py

Offline sweep of ball-detection settings against ground truth.

Inference runs ONCE per image size at a very low confidence. Confidence
thresholds and "keep only the top-k boxes per frame" post-processing are then
swept in memory, so a full sweep costs one inference pass per image size.

Reports
  * precision / recall / F1 at IoU>=0.5 AND with a centre-distance criterion
    (the one that matters for tracking: is the ball found, even if the box is
    a few pixels off?)
  * false positives split into "empty frame" (no GT ball in the image) and
    "extra" (additional box in a frame that does have a GT ball)
  * recall by ball size (scaled to a 640px image so mixed-resolution data is
    comparable) and by image-resolution group (160 / 320 / 640 px)
  * "reachable" recall: the fraction of GT balls that ANY candidate lands on
    at very low confidence. This is the ceiling for thresholds / trackers.

Writes (to runs/diagnostics/ball_sweep/<split>/)
  sweep_results.csv, per_gt.csv (feeds ball_review_sheet.py)

Usage
  python ball_eval_sweep.py
  python ball_eval_sweep.py --split test --imgsz 640 960 1280
  python ball_eval_sweep.py --weights path/to/best.pt --nms-iou 0.5
"""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "V1"
DEFAULT_WEIGHTS = (
    ROOT / "runs" / "football" / "football_detection_v1-4" / "weights" / "best.pt"
)
OUT_ROOT = ROOT / "runs" / "diagnostics" / "ball_sweep"

BALL_CLASS_ID = 1
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CONF_GRID = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
TOPK_MODES = {"all": None, "top1": 1}

MATCH_IOU = 0.50            # IoU criterion
CENTRE_TOL_FACTOR = 1.0     # centre must be within 1.0 x GT longer side ...
CENTRE_TOL_MIN_PX = 4.0     # ... but never tighter than this many pixels

SIZE_EDGES = [0, 8, 16, 32, 64, 1e9]   # sqrt(w*h) of GT box, scaled to a 640px image
GROUP_SIDES = (160, 320, 640)           # image resolutions found in this dataset

SUSPECT_MAX_AREA_FRAC = 0.10   # GT box covering >10% of the image
SUSPECT_MAX_ASPECT = 2.0       # long side / short side
SUSPECT_MIN_SIDE_PX = 3.0      # shorter side below this many pixels


# ----------------------------------------------------------------------------
# geometry / matching
# ----------------------------------------------------------------------------

def load_gt(label_path, width, height):
    """Ball boxes of one label file as pixel xyxy."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        v = line.split()
        if len(v) != 5:
            continue
        try:
            cls, xc, yc, bw, bh = (float(x) for x in v)
        except ValueError:
            continue
        if int(cls) != BALL_CLASS_ID:
            continue
        boxes.append([
            (xc - bw / 2) * width, (yc - bh / 2) * height,
            (xc + bw / 2) * width, (yc + bh / 2) * height,
        ])
    return boxes


def iou_xyxy(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def centre_dist(a, b):
    return math.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2,
                      (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2)


def centre_tol(gt_box):
    longest = max(gt_box[2] - gt_box[0], gt_box[3] - gt_box[1])
    return max(CENTRE_TOL_MIN_PX, CENTRE_TOL_FACTOR * longest)


def greedy_match(preds, gts, criterion):
    """
    preds: (n, 5) array sorted by confidence, descending.
    Returns, per prediction, the index of the GT it matched (or -1).
    """
    used = set()
    assigned = []
    for p in preds:
        best, best_score = -1, None
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            if criterion == "iou":
                score = iou_xyxy(p[:4], g)
                ok = score >= MATCH_IOU
            else:
                d = centre_dist(p[:4], g)
                score = -d
                ok = d <= centre_tol(g)
            if ok and (best_score is None or score > best_score):
                best, best_score = gi, score
        if best >= 0:
            used.add(best)
        assigned.append(best)
    return assigned


def evaluate(stems, preds, gt, conf_thr, topk):
    """
    One operating point (confidence threshold + keep-top-k) over all images.
    Returns (counters, hits) where hits[(stem, gt_index, criterion)] is a bool.
    """
    tot = Counter()
    hits = {}
    for stem in stems:
        p = preds[stem]
        p = p[p[:, 4] >= conf_thr]
        if topk is not None:
            p = p[:topk]
        g = gt[stem]
        for crit in ("iou", "centre"):
            assigned = greedy_match(p, g, crit)
            tp = sum(a >= 0 for a in assigned)
            fp = len(assigned) - tp
            tot[f"tp_{crit}"] += tp
            tot[f"fp_{crit}"] += fp
            tot[f"fn_{crit}"] += len(g) - tp
            matched = {a for a in assigned if a >= 0}
            for gi in range(len(g)):
                hits[(stem, gi, crit)] = gi in matched
            if crit == "iou":
                tot["fp_extra" if g else "fp_empty"] += fp
    return tot, hits


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def make_row(imgsz, mode, conf, tot):
    p_i, r_i, f_i = prf(tot["tp_iou"], tot["fp_iou"], tot["fn_iou"])
    p_c, r_c, f_c = prf(tot["tp_centre"], tot["fp_centre"], tot["fn_centre"])
    return {
        "imgsz": imgsz, "mode": mode, "conf": conf,
        "P_iou": p_i, "R_iou": r_i, "F1_iou": f_i,
        "P_ctr": p_c, "R_ctr": r_c, "F1_ctr": f_c,
        "fp_empty": tot["fp_empty"], "fp_extra": tot["fp_extra"],
        "n_pred": tot["tp_iou"] + tot["fp_iou"],
    }


# ----------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------

def run_inference(model, images, imgsz, nms_iou, device, augment, raw_conf, batch):
    """Ball candidates per image, sorted by confidence, original-pixel xyxy.

    NOTE: Ultralytics treats a LIST of paths as ONE batch, so a list of the whole
    split would push every image through the network at once (CUDA OOM). We feed
    it small chunks instead.
    """
    preds = {}
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=imgsz,
            conf=raw_conf,
            iou=nms_iou,
            device=device,
            classes=[BALL_CLASS_ID],
            augment=augment,
            verbose=False,
        )
        for result in results:
            stem = Path(result.path).stem
            if result.boxes is not None and len(result.boxes):
                xyxy = result.boxes.xyxy.cpu().numpy()
                conf = result.boxes.conf.cpu().numpy()[:, None]
                arr = np.concatenate([xyxy, conf], axis=1)
                arr = arr[np.argsort(-arr[:, 4])]
            else:
                arr = np.zeros((0, 5))
            preds[stem] = arr
        done = min(start + batch, len(images))
        if (start // batch) % 50 == 0 or done == len(images):
            print(f"  {done}/{len(images)} images")
    return preds


# ----------------------------------------------------------------------------
# reporting helpers
# ----------------------------------------------------------------------------

def bucket_label(size_px):
    for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:]):
        if lo <= size_px < hi:
            return f"{lo:g}-{hi:g}px" if hi < 1e8 else f">={lo:g}px"
    return "?"


def suspect_reason(row, area_limit):
    reasons = []
    if row["area_frac"] > area_limit:
        reasons.append("huge")
    if row["aspect"] > SUSPECT_MAX_ASPECT:
        reasons.append("odd_aspect")
    if min(row["w_px"], row["h_px"]) < SUSPECT_MIN_SIDE_PX:
        reasons.append("tiny")
    return "|".join(reasons)


def print_sweep_table(rows):
    print("       |      IoU>=0.5       |       centre        |  false positives | ")
    print(f"{'conf':>6} | {'P':>5} {'R':>5} {'F1':>5} | {'P':>5} {'R':>5} {'F1':>5} "
          f"| {'empty':>7} {'extra':>7} | {'preds':>6}")
    for r in rows:
        print(f"{r['conf']:>6.2f} | {r['P_iou']:>5.3f} {r['R_iou']:>5.3f} {r['F1_iou']:>5.3f} "
              f"| {r['P_ctr']:>5.3f} {r['R_ctr']:>5.3f} {r['F1_ctr']:>5.3f} "
              f"| {r['fp_empty']:>7d} {r['fp_extra']:>7d} | {r['n_pred']:>6d}")


def print_size_table(gt_rows, hits_op, hits_reach):
    buckets = defaultdict(lambda: [0, 0, 0, 0])
    for r in gt_rows:
        key = (r["stem"], r["gi"])
        b = buckets[bucket_label(r["size_640"])]
        b[0] += 1
        b[1] += hits_op[key + ("iou",)]
        b[2] += hits_op[key + ("centre",)]
        b[3] += hits_reach[key + ("centre",)]
    print(f"{'GT ball size':>12} | {'n':>5} | {'R(IoU)':>7} {'R(centre)':>9} | {'reachable':>9}")
    order = [bucket_label(lo) for lo in SIZE_EDGES[:-1]]
    for name in order:
        if name not in buckets:
            continue
        n, hi, hc, hr = buckets[name]
        print(f"{name:>12} | {n:>5d} | {hi / n:>7.3f} {hc / n:>9.3f} | {hr / n:>9.3f}")


def group_label(long_side):
    return f"{long_side}px imgs" if long_side in GROUP_SIDES else "other sizes"


def print_group_table(gt_rows, hits_op, hits_reach):
    groups = defaultdict(lambda: [0, 0, 0, 0])
    for r in gt_rows:
        key = (r["stem"], r["gi"])
        g = groups[group_label(r["long_side"])]
        g[0] += 1
        g[1] += hits_op[key + ("iou",)]
        g[2] += hits_op[key + ("centre",)]
        g[3] += hits_reach[key + ("centre",)]
    print(f"{'image group':>12} | {'n':>5} | {'R(IoU)':>7} {'R(centre)':>9} | {'reachable':>9}")
    for name in [group_label(side) for side in GROUP_SIDES] + ["other sizes"]:
        if name not in groups:
            continue
        n, hi, hc, hr = groups[name]
        print(f"{name:>12} | {n:>5d} | {hi / n:>7.3f} {hc / n:>9.3f} | {hr / n:>9.3f}")


def nearest_candidate(preds, gt_box):
    """Return the candidate nearest to a GT centre, with IoU and distance."""
    if len(preds) == 0:
        return None
    ranked = []
    for pred in preds:
        box = pred[:4]
        ranked.append((centre_dist(box, gt_box), -iou_xyxy(box, gt_box), pred))
    distance, neg_iou, pred = min(ranked, key=lambda item: (item[0], item[1]))
    return {
        "x1": float(pred[0]), "y1": float(pred[1]),
        "x2": float(pred[2]), "y2": float(pred[3]),
        "conf": float(pred[4]), "iou": float(-neg_iou),
        "centre_dist": float(distance),
        "centre_hit": int(distance <= centre_tol(gt_box)),
    }


def add_candidate_fields(row, prefix, candidate):
    """Append prediction coordinates and quality fields to a GT row."""
    for name in ("x1", "y1", "x2", "y2", "conf", "iou", "centre_dist", "centre_hit"):
        value = None if candidate is None else candidate[name]
        row[f"{prefix}_{name}"] = round(value, 4) if isinstance(value, float) else value


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280])
    ap.add_argument("--nms-iou", type=float, default=0.5,
                    help="NMS IoU passed to predict (only used by NMS-based inference)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8,
                    help="images per forward pass (lower it if you hit CUDA OOM)")
    ap.add_argument("--augment", action="store_true",
                    help="test-time augmentation (not supported by every model type)")
    ap.add_argument("--suspect-area", type=float, default=SUSPECT_MAX_AREA_FRAC,
                    help="flag GT boxes covering more than this fraction of the image")
    return ap.parse_args()


def main():
    args = parse_args()

    image_dir = DATASET / "images" / args.split
    label_dir = DATASET / "labels" / args.split
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {image_dir}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing weights: {args.weights}")

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    stems = [p.stem for p in images]
    raw_conf = min(CONF_GRID)

    # ---- ground truth -------------------------------------------------
    gt, gt_rows, img_sizes = {}, [], Counter()
    for p in images:
        with Image.open(p) as im:
            w, h = im.size
        img_sizes[(w, h)] += 1
        boxes = load_gt(label_dir / f"{p.stem}.txt", w, h)
        gt[p.stem] = boxes
        for gi, b in enumerate(boxes):
            bw, bh = b[2] - b[0], b[3] - b[1]
            row = {
                "image": p.name, "stem": p.stem, "gi": gi,
                "img_w": w, "img_h": h,
                "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                "w_px": bw, "h_px": bh,
                "size_px": math.sqrt(max(bw * bh, 0.0)),
                "size_640": math.sqrt(max(bw * bh, 0.0)) * 640 / max(w, h),
                "long_side": max(w, h),
                "aspect": max(bw, bh) / max(min(bw, bh), 1e-6),
                "area_frac": (bw * bh) / (w * h),
            }
            row["suspect"] = suspect_reason(row, args.suspect_area)
            gt_rows.append(row)

    n_gt = len(gt_rows)
    if n_gt == 0:
        raise RuntimeError("No ground-truth balls found in this split.")
    sizes = np.array([r["size_px"] for r in gt_rows])
    print(f"\nSplit: {args.split} | images: {len(images)} | GT balls: {n_gt} "
          f"| images with a GT ball: {sum(1 for s in stems if gt[s])}")
    print("Most common image sizes:", img_sizes.most_common(3))
    s640 = np.array([r["size_640"] for r in gt_rows])
    frac = np.array([r["area_frac"] for r in gt_rows])
    print("GT ball size, original px:   "
          f"p10={np.percentile(sizes, 10):.1f}  median={np.median(sizes):.1f}  "
          f"p90={np.percentile(sizes, 90):.1f}  max={sizes.max():.1f}")
    print("GT ball size, 640px-scaled:  "
          f"p10={np.percentile(s640, 10):.1f}  median={np.median(s640):.1f}  "
          f"p90={np.percentile(s640, 90):.1f}  max={s640.max():.1f}")
    print("Ball box as share of image area: "
          + "  ".join(f">{t:.0%}: {int((frac > t).sum())}" for t in (0.02, 0.05, 0.10, 0.25)))
    print(f"Suspect labels flagged: {sum(1 for r in gt_rows if r['suspect'])} "
          f"(huge > {args.suspect_area:.0%} of image, aspect > {SUSPECT_MAX_ASPECT:g}, "
          f"side < {SUSPECT_MIN_SIDE_PX:g}px)")

    # ---- inference + sweep -------------------------------------------
    from ultralytics import YOLO  # imported late so --help works anywhere
    model = YOLO(str(args.weights))

    results, cache = [], {}
    for imgsz in args.imgsz:
        print(f"\nInference at imgsz={imgsz}")
        cache[imgsz] = run_inference(
            model, images, imgsz, args.nms_iou, args.device, args.augment, raw_conf,
            args.batch,
        )
        for mode, topk in TOPK_MODES.items():
            block = []
            for conf in CONF_GRID:
                tot, _ = evaluate(stems, cache[imgsz], gt, conf, topk)
                block.append(make_row(imgsz, mode, conf, tot))
            results.extend(block)
            print(f"\n--- imgsz={imgsz} | keep={mode} ---")
            print_sweep_table(block)

    # ---- summary ------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY: best confidence per setting (by F1 at IoU>=0.5)")
    print("=" * 78)
    print(f"{'imgsz':>6} {'keep':>5} {'conf':>5} | {'P':>5} {'R':>5} {'F1':>5} | "
          f"{'R(centre)':>9} | {'reachable':>9}")
    best_by_setting = {}
    for imgsz in args.imgsz:
        reach = next(r for r in results
                     if r["imgsz"] == imgsz and r["mode"] == "all" and r["conf"] == raw_conf)
        for mode in TOPK_MODES:
            rows = [r for r in results if r["imgsz"] == imgsz and r["mode"] == mode]
            best = max(rows, key=lambda r: r["F1_iou"])
            best_by_setting[(imgsz, mode)] = best
            print(f"{imgsz:>6d} {mode:>5} {best['conf']:>5.2f} | {best['P_iou']:>5.3f} "
                  f"{best['R_iou']:>5.3f} {best['F1_iou']:>5.3f} | {best['R_ctr']:>9.3f} "
                  f"| {reach['R_ctr']:>9.3f}")
    print("reachable = share of GT balls with ANY candidate on them at "
          f"conf={raw_conf:g} (centre criterion): the ceiling for thresholds/trackers.")

    # ---- recall by ball size -----------------------------------------
    for imgsz in args.imgsz:
        best = best_by_setting[(imgsz, "top1")]
        _, hits_op = evaluate(stems, cache[imgsz], gt, best["conf"], 1)
        _, hits_reach = evaluate(stems, cache[imgsz], gt, raw_conf, None)
        print(f"\n--- recall by GT ball size (640px-scaled) | imgsz={imgsz} "
              f"| top1 @ conf {best['conf']:.2f} ---")
        print_size_table(gt_rows, hits_op, hits_reach)
        print(f"\n--- recall by image resolution | imgsz={imgsz} ---")
        print_group_table(gt_rows, hits_op, hits_reach)

    # ---- files --------------------------------------------------------
    out_dir = OUT_ROOT / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "sweep_results.csv",
              [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
               for r in results])

    primary_imgsz, primary = max(
        ((i, best_by_setting[(i, "top1")]) for i in args.imgsz),
        key=lambda t: t[1]["F1_iou"],
    )
    _, hits_op = evaluate(stems, cache[primary_imgsz], gt, primary["conf"], 1)
    _, hits_reach = evaluate(stems, cache[primary_imgsz], gt, raw_conf, None)
    per_gt = []
    for r in gt_rows:
        key = (r["stem"], r["gi"])
        out = dict(r)
        out["hit_iou"] = int(hits_op[key + ("iou",)])
        out["hit_centre"] = int(hits_op[key + ("centre",)])
        out["reachable_centre"] = int(hits_reach[key + ("centre",)])

        gt_box = gt[r["stem"]][r["gi"]]
        raw_candidate = nearest_candidate(cache[primary_imgsz][r["stem"]], gt_box)
        add_candidate_fields(out, "candidate", raw_candidate)

        op_preds = cache[primary_imgsz][r["stem"]]
        op_preds = op_preds[op_preds[:, 4] >= primary["conf"]][:1]
        op_candidate = nearest_candidate(op_preds, gt_box)
        add_candidate_fields(out, "operating", op_candidate)

        per_gt.append({k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in out.items()})
    write_csv(out_dir / "per_gt.csv", per_gt)

    print(f"\nPer-GT table uses imgsz={primary_imgsz}, top1, conf={primary['conf']:.2f}.")
    print(f"Saved: {out_dir / 'sweep_results.csv'}")
    print(f"Saved: {out_dir / 'per_gt.csv'}  (next: python ball_review_sheet.py --split {args.split})")


if __name__ == "__main__":
    main()