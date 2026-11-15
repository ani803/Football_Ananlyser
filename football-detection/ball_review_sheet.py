"""Create ball review contact sheets with GT and prediction overlays.

Requires the enhanced per_gt.csv produced by ball_eval_sweep_review.py.
GT box = green; nearest raw candidate = red; operating-point candidate = cyan.
"""
import argparse
import csv
import random
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "V1"
SWEEP_ROOT = ROOT / "runs" / "diagnostics" / "ball_sweep"
OUT_ROOT = ROOT / "runs" / "diagnostics" / "ball_review"
TILE, CAPTION_H, MIN_WINDOW_PX = 288, 42, 96


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def crop_square(img, cx, cy, half):
    h, w = img.shape[:2]
    x1, y1 = int(round(cx-half)), int(round(cy-half))
    x2, y2 = int(round(cx+half)), int(round(cy+half))
    crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    if crop.size == 0:
        return None, x1, y1
    crop = cv2.copyMakeBorder(crop, max(0,-y1), max(0,y2-h),
                               max(0,-x1), max(0,x2-w),
                               cv2.BORDER_CONSTANT, value=(0,0,0))
    return crop, x1, y1


def parse_box(row, prefix):
    try:
        box = [float(row[f"{prefix}_{k}"]) for k in ("x1","y1","x2","y2")]
        if not all(np.isfinite(v) for v in box):
            return None
        return box
    except (KeyError, TypeError, ValueError):
        return None


def draw_box(tile, box, ox, oy, scale, color, label):
    if box is None:
        return False
    x1,y1,x2,y2 = box
    h,w = tile.shape[:2]
    a,b = int(round((x1-ox)*scale)), int(round((y1-oy)*scale))
    c,d = int(round((x2-ox)*scale)), int(round((y2-oy)*scale))
    if c < 0 or d < 0 or a >= w or b >= h:
        return False
    p1, p2 = (max(0,a),max(0,b)), (min(w-1,c),min(h-1,d))
    cv2.rectangle(tile,p1,p2,color,2)
    cv2.putText(tile,label,(p1[0],max(12,p1[1]-4)),cv2.FONT_HERSHEY_SIMPLEX,
                0.38,color,1,cv2.LINE_AA)
    return True


def metric(row, prefix, name):
    try:
        return f"{float(row[f'{prefix}_{name}']):.2f}"
    except (KeyError,TypeError,ValueError):
        return "n/a"


def make_tile(image_path,row):
    img=cv2.imread(str(image_path))
    if img is None:
        return None
    gt=[float(row[k]) for k in ("x1","y1","x2","y2")]
    raw=parse_box(row,"candidate")
    operating=parse_box(row,"operating")
    x1,y1,x2,y2=gt
    half=max(MIN_WINDOW_PX,3*max(x2-x1,y2-y1))/2
    crop,ox,oy=crop_square(img,(x1+x2)/2,(y1+y2)/2,half)
    if crop is None:
        return None
    scale=TILE/crop.shape[0]
    tile=cv2.resize(crop,(TILE,TILE),interpolation=cv2.INTER_CUBIC)
    draw_box(tile,gt,ox,oy,scale,(0,255,0),"GT")
    raw_drawn=draw_box(tile,raw,ox,oy,scale,(0,0,255),"raw candidate")
    draw_box(tile,operating,ox,oy,scale,(255,255,0),"operating")

    status=("hit" if row.get("hit_centre")=="1" else
            "reachable" if row.get("reachable_centre")=="1" else "unreachable")
    caption=np.zeros((CAPTION_H,TILE,3),dtype=np.uint8)
    line1=f"{row['stem'][-18:]} {float(row['size_px']):.1f}px {status}"
    line2=f"raw conf={metric(row,'candidate','conf')} IoU={metric(row,'candidate','iou')}"
    line3=f"operating conf={metric(row,'operating','conf')}"
    for text,y,color in ((line1,13,(255,255,255)),(line2,27,(220,220,220)),(line3,40,(220,220,220))):
        cv2.putText(caption,text[:48],(3,y),cv2.FONT_HERSHEY_SIMPLEX,0.32,color,1,cv2.LINE_AA)
    if raw is not None and not raw_drawn:
        cv2.putText(caption,"raw candidate outside crop",(3,41),cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,(0,0,255),1,cv2.LINE_AA)
    return np.vstack([caption,tile])


def build_sheets(name,rows,image_dir,out_dir,cols,rows_per_sheet):
    per_sheet=cols*rows_per_sheet
    tiles=[t for r in rows if (t:=make_tile(image_dir/r["image"],r)) is not None]
    if not tiles:
        print(f"{name:14s}: nothing to show")
        return
    blank=np.zeros_like(tiles[0]); count=0
    for start in range(0,len(tiles),per_sheet):
        chunk=tiles[start:start+per_sheet]+[blank]*max(0,per_sheet-len(tiles[start:start+per_sheet]))
        grid=np.vstack([np.hstack(chunk[i*cols:(i+1)*cols]) for i in range(rows_per_sheet)])
        count+=1
        cv2.imwrite(str(out_dir/f"{name}_{count:02d}.jpg"),grid,[cv2.IMWRITE_JPEG_QUALITY,92])
    print(f"{name:14s}: {len(tiles)} tiles -> {count} sheet(s)")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split",default="val",choices=["train","val","test"])
    ap.add_argument("--cols",type=int,default=4)
    ap.add_argument("--rows",type=int,default=3)
    ap.add_argument("--max-tiles",type=int,default=96)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    csv_path=SWEEP_ROOT/args.split/"per_gt.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run ball_eval_sweep_review.py first: {csv_path}")
    rows=read_rows(csv_path)
    if not rows or "candidate_conf" not in rows[0]:
        raise RuntimeError("per_gt.csv lacks prediction fields; rerun ball_eval_sweep_review.py")
    image_dir=DATASET/"images"/args.split
    out_dir=OUT_ROOT/args.split
    out_dir.mkdir(parents=True,exist_ok=True)
    rng=random.Random(args.seed)
    suspects=[r for r in rows if r["suspect"]]
    suspects.sort(key=lambda r:float(r["area_frac"]),reverse=True)
    unreachable=[r for r in rows if r["hit_centre"]=="0" and r["reachable_centre"]=="0"]
    ranked_out=[r for r in rows if r["hit_centre"]=="0" and r["reachable_centre"]=="1"]
    rng.shuffle(unreachable); rng.shuffle(ranked_out)
    print(f"GT balls: {len(rows)} | suspect: {len(suspects)} | unreachable: {len(unreachable)} | ranked-out/low-conf: {len(ranked_out)}")
    for name,group in (("suspect",suspects),("unreachable",unreachable),("ranked_out",ranked_out)):
        build_sheets(name,group[:args.max_tiles],image_dir,out_dir,args.cols,args.rows)
    print(f"\nSheets saved to: {out_dir}")

if __name__=="__main__":
    main()
