#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = row.get("pred_rel_err", "")
            actual = row.get("actual_sum_rel_err", "")
            if not pred or pred == "None":
                continue
            rows.append(
                {
                    "target": row["target"],
                    "pred": float(pred),
                    "actual": float(actual),
                }
            )
    return rows


def log10_safe(x: float) -> float:
    return math.log10(max(x, 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="compare.csv path")
    ap.add_argument("--out", required=True, help="output png path")
    ap.add_argument("--title", default="Predicted vs Actual Sum Relative Error")
    args = ap.parse_args()

    rows = load_rows(Path(args.csv))
    if not rows:
        raise SystemExit("no usable rows in csv")

    xs = [log10_safe(r["pred"]) for r in rows]
    ys = [log10_safe(r["actual"]) for r in rows]

    lo = min(xs + ys)
    hi = max(xs + ys)
    pad = 0.4
    lo -= pad
    hi += pad

    W, H = 1200, 900
    L, R, T, B = 120, 60, 90, 110
    PW, PH = W - L - R, H - T - B

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    def sx(v: float) -> float:
        return L + (v - lo) / (hi - lo) * PW

    def sy(v: float) -> float:
        return T + (hi - v) / (hi - lo) * PH

    # Axes and grid
    draw.rectangle([L, T, W - R, H - B], outline="black", width=2)
    ticks = list(range(math.floor(lo), math.ceil(hi) + 1))
    for t in ticks:
        x = sx(float(t))
        y = sy(float(t))
        draw.line([x, T, x, H - B], fill=(230, 230, 230), width=1)
        draw.line([L, y, W - R, y], fill=(230, 230, 230), width=1)
        label = f"1e{t}"
        draw.text((x - 12, H - B + 10), label, fill="black", font=font)
        draw.text((20, y - 6), label, fill="black", font=font)

    # y = x reference
    draw.line([sx(lo), sy(lo), sx(hi), sy(hi)], fill=(120, 120, 120), width=2)

    palette = {
        "8": (214, 39, 40),
        "16": (255, 127, 14),
        "32": (44, 160, 44),
    }

    for r in rows:
        x = sx(log10_safe(r["pred"]))
        y = sy(log10_safe(r["actual"]))
        bits = r["target"].split("_")[1]
        color = palette.get(bits, (31, 119, 180))
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color, outline="black")
        draw.text((x + 8, y - 8), r["target"], fill=color, font=font)

    draw.text((W // 2 - 140, 25), args.title, fill="black", font=font)
    draw.text((W // 2 - 80, H - 40), "Predicted relative error (log10 scale)", fill="black", font=font)
    draw.text((10, T - 20), "Actual sum relative error (log10 scale)", fill="black", font=font)

    # Legend
    legend_x = W - R - 180
    legend_y = T + 15
    draw.rectangle([legend_x - 10, legend_y - 10, legend_x + 150, legend_y + 70], outline="black", width=1)
    for i, (bits, color) in enumerate([("8", palette["8"]), ("16", palette["16"]), ("32", palette["32"])]):
        yy = legend_y + i * 22
        draw.ellipse([legend_x, yy, legend_x + 10, yy + 10], fill=color, outline="black")
        draw.text((legend_x + 18, yy - 2), f"posit_{bits}_*", fill="black", font=font)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
