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
            if not pred or pred == "None" or not actual or actual == "None":
                continue
            rows.append(
                {
                    "target": row["target"],
                    "pred": float(pred),
                    "actual": float(actual),
                }
            )
    return rows


def load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def value_for_plot(v: float, log_scale: bool) -> float:
    return math.log10(max(v, 1e-12)) if log_scale else v


def fmt_tick(v: float, log_scale: bool) -> str:
    return f"1e{int(v)}" if log_scale else f"{v:.2g}"


def fmt_value(v: float) -> str:
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e3 or av < 1e-2:
        return f"{v:.2e}"
    if av >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"


def draw_boxed_text(draw, xy, text, font, text_fill, box_fill=(255, 255, 255), box_outline=(0, 0, 0)):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x, y = xy
    pad_x = 6
    pad_y = 4
    draw.rounded_rectangle(
        [x, y, x + tw + 2 * pad_x, y + th + 2 * pad_y],
        radius=6,
        fill=box_fill,
        outline=box_outline,
        width=1,
    )
    draw.text((x + pad_x, y + pad_y), text, fill=text_fill, font=font)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="compare.csv path")
    ap.add_argument("--out", required=True, help="output png path")
    ap.add_argument("--title", default="Predicted vs Actual Sum Relative Error")
    ap.add_argument("--log-scale", action="store_true")
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--title-font-size", type=int, default=34)
    ap.add_argument("--label-font-size", type=int, default=26)
    ap.add_argument("--width", type=int, default=1800)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--show-values", action="store_true")
    ap.add_argument("--show-table", action="store_true")
    args = ap.parse_args()

    rows = load_rows(Path(args.csv))
    if not rows:
        raise SystemExit("no usable rows in csv")

    pred_vals = [value_for_plot(r["pred"], args.log_scale) for r in rows]
    actual_vals = [value_for_plot(r["actual"], args.log_scale) for r in rows]
    hi = max(pred_vals + actual_vals)
    lo = min(0.0, min(pred_vals + actual_vals))
    if hi <= lo:
        hi = lo + 1.0
    hi += 0.08 * (hi - lo)

    W, H = args.width, args.height
    extra_table_h = 0
    if args.show_table:
        extra_table_h = max(320, 54 * (len(rows) + 2))
        H += extra_table_h
    L, R, T, B = 140, 60, 100, 220 + extra_table_h
    PW, PH = W - L - R, H - T - B

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    font = load_font(args.font_size)
    title_font = load_font(args.title_font_size)
    label_font = load_font(args.label_font_size)

    def sy(v: float) -> float:
        return T + (hi - v) / (hi - lo) * PH

    draw.rectangle([L, T, W - R, H - B], outline="black", width=2)

    tick_count = 6
    if args.log_scale:
        ticks = list(range(math.floor(lo), math.ceil(hi) + 1))
    else:
        step = (hi - lo) / tick_count
        ticks = [lo + i * step for i in range(tick_count + 1)]
    for tv in ticks:
        y = sy(float(tv))
        draw.line([L, y, W - R, y], fill=(230, 230, 230), width=1)
        label = fmt_tick(float(tv), args.log_scale)
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((L - 18 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2), label, fill="black", font=font)

    group_w = PW / max(len(rows), 1)
    bar_w = min(36, group_w * 0.28)
    pred_color = (52, 101, 164)
    actual_color = (204, 0, 0)

    for i, row in enumerate(rows):
        cx = L + group_w * (i + 0.5)
        pred_v = value_for_plot(row["pred"], args.log_scale)
        actual_v = value_for_plot(row["actual"], args.log_scale)

        pred_x0 = cx - bar_w - 4
        pred_x1 = cx - 4
        actual_x0 = cx + 4
        actual_x1 = cx + bar_w + 4

        base_y = sy(lo)
        draw.rectangle([pred_x0, sy(pred_v), pred_x1, base_y], fill=pred_color, outline="black")
        draw.rectangle([actual_x0, sy(actual_v), actual_x1, base_y], fill=actual_color, outline="black")

        if args.show_values:
            pred_text = fmt_value(row["pred"])
            actual_text = fmt_value(row["actual"])
            pred_bbox = draw.textbbox((0, 0), pred_text, font=font)
            actual_bbox = draw.textbbox((0, 0), actual_text, font=font)
            pred_w = (pred_bbox[2] - pred_bbox[0]) + 12
            actual_w = (actual_bbox[2] - actual_bbox[0]) + 12
            pred_x = max(L + 4, min(pred_x0 - 20, W - R - pred_w - 4))
            actual_x = max(L + 4, min(actual_x0 + 20, W - R - actual_w - 4))
            pred_y = max(T + 8, sy(pred_v) - 36)
            actual_y = max(T + 8, sy(actual_v) - 36)
            draw_boxed_text(draw, (pred_x, pred_y), pred_text, font, pred_color)
            draw_boxed_text(draw, (actual_x, actual_y), actual_text, font, actual_color)

        label = row["target"]
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        tx = cx - tw / 2
        ty = H - B + 20
        label_img = Image.new("RGBA", (tw + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_img)
        label_draw.text((4, 4), label, fill="black", font=font)
        label_img = label_img.rotate(35, expand=True, fillcolor=(255, 255, 255, 0))
        img.paste(label_img, (int(tx), int(ty)), label_img)

    title_bbox = draw.textbbox((0, 0), args.title, font=title_font)
    draw.text(((W - (title_bbox[2] - title_bbox[0])) / 2, 24), args.title, fill="black", font=title_font)

    y_label = "Relative error (log10 scale)" if args.log_scale else "Relative error"
    draw.text((L, H - 70), "Target format", fill="black", font=label_font)
    draw.text((16, T - 48), y_label, fill="black", font=label_font)

    legend_x = W - R - 300
    legend_y = T + 20
    draw.rectangle([legend_x, legend_y, legend_x + 250, legend_y + 100], outline="black", width=1)
    draw.rectangle([legend_x + 18, legend_y + 18, legend_x + 48, legend_y + 48], fill=pred_color, outline="black")
    draw.text((legend_x + 62, legend_y + 12), "Predicted", fill="black", font=font)
    draw.rectangle([legend_x + 18, legend_y + 58, legend_x + 48, legend_y + 88], fill=actual_color, outline="black")
    draw.text((legend_x + 62, legend_y + 52), "Actual sum rel err", fill="black", font=font)

    if args.show_table:
        table_top = H - extra_table_h + 20
        table_left = L
        table_right = W - R
        row_h = 48
        col_target = 260
        col_pred = 350
        col_actual = 350
        x0 = table_left
        x1 = x0 + col_target
        x2 = x1 + col_pred
        x3 = min(table_right, x2 + col_actual)

        draw.rectangle([x0, table_top, x3, table_top + row_h], fill=(240, 240, 240), outline="black", width=1)
        draw.text((x0 + 12, table_top + 10), "Target", fill="black", font=font)
        draw.text((x1 + 12, table_top + 10), "Predicted", fill="black", font=font)
        draw.text((x2 + 12, table_top + 10), "Actual sum rel err", fill="black", font=font)

        for i, row in enumerate(rows):
            y = table_top + row_h * (i + 1)
            fill = (255, 255, 255) if i % 2 == 0 else (248, 248, 248)
            draw.rectangle([x0, y, x3, y + row_h], fill=fill, outline="black", width=1)
            draw.line([x1, y, x1, y + row_h], fill="black", width=1)
            draw.line([x2, y, x2, y + row_h], fill="black", width=1)
            draw.text((x0 + 12, y + 10), row["target"], fill="black", font=font)
            draw.text((x1 + 12, y + 10), fmt_value(row["pred"]), fill=pred_color, font=font)
            draw.text((x2 + 12, y + 10), fmt_value(row["actual"]), fill=actual_color, font=font)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
