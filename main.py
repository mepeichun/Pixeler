# main.py
from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import KMeans
import json
from scipy.spatial import KDTree
from collections import Counter
import math

from utils import (
    hex_to_rgb, rgb_to_hex, pick_text_color,
    load_font, text_size, fit_font_to_cell
)

# =========================
# 可调参数
# =========================
IMG_PATH = "./images/usagi.png"      # 输入图片
PALETTES_JSON = "./assets/palettes.json"
PALETTE_NAME = "mard"       # 使用的调色板名称，如 "mard" / "hama" /, 更多请查看./assets中的palettes.json
OUTPUT_SIZE = 38            # 输出像素格尺寸（宽高各 OUTPUT_SIZE 个格）
COLOR_NUM = 12              # KMeans 聚类数（期望的颜色数，一般情况，实际输出的拼豆颜色会比这个期待值小一点）
ADD_COORD = True            # 聚类时是否考虑坐标
LAMBDA = 0.15               # 坐标权重(0~1)
OUT_IMAGE = "./output/pixelated_grid_with_legend.png" # 输出文件名


# =========================
# 以下参数通常无需调整
# =========================

# 可视化细节（主格子）
CELL_PX = 36                # 每个像素格的可视化尺寸(像素)
GRID_THICKNESS = 1          # 网格线粗细
DRAW_GRID = True            # 是否画网格

# 文字与字体
LABEL_FONT_PATH = "./assets/Roboto-VariableFont_wdth,wght.ttf"  # None 则用默认字体
LABEL_FONT_SCALE = 0.45      # 单格内色号文字大小，相对 CELL_PX 的比例(0.4~0.7 较合适)
LABEL_MAX_WIDTH_RATIO = 0.7 # 文本最大宽度占单格宽度比例(防止溢出)
AXIS_FONT_SCALE = 0.5       # 坐标轴编号字体大小(相对 CELL_PX)

# 图例布局（放在整体 grid 下侧）
LEGEND_SWATCH_W = 48
LEGEND_SWATCH_H = 24
LEGEND_GAP_X = 24
LEGEND_GAP_Y = 16
LEGEND_COLS = 3             # 图例列数
LEGEND_TEXT_SIZE = 16
LEGEND_TITLE_SIZE = 18
LEGEND_TOP_GAP = 28         # grid 与 legend 的垂直间距

# 留白（相对之前更宽裕）
OUTER_PAD = 56              # 画布外围留白
AXIS_TOP_BAND = None        # 顶部坐标带高度（None=自动按字体估算）
AXIS_LEFT_BAND = None       # 左侧坐标带宽度（None=自动按字体估算）
INNER_GAP = 12              # 坐标带与 grid 之间的小间距

# 透明像素处理
ALPHA_IGNORE_THRESHOLD = 51   # <=51 的 alpha 视为“忽略（不填豆）”


# =========================
# 功能函数（主流程）
# =========================
def load_and_pixelate(path: str, out_size: int, alpha_ignore_th: int):
    img = Image.open(path).convert("RGBA")
    resized = img.resize((out_size * 2, out_size * 2), Image.BILINEAR)
    arr = np.array(resized, dtype=np.uint8)  # (H2, W2, 4)
    H2, W2 = arr.shape[:2]
    height = width = out_size

    pixel_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    ignored_mask = np.zeros((height, width), dtype=bool)

    for y in range(height):
        for x in range(width):
            block = arr[y * 2:y * 2 + 2, x * 2:x * 2 + 2, :]  # (2,2,4)
            alphas = block[..., 3].astype(np.float32)
            valid = alphas > alpha_ignore_th
            if not np.any(valid):
                ignored_mask[y, x] = True
                pixel_rgb[y, x] = (255, 255, 255)
                continue
            a = alphas[valid] / 255.0
            rgb = block[..., :3][valid].astype(np.float32)
            weighted = (rgb * a[:, None]).sum(axis=0) / a.sum()
            pixel_rgb[y, x] = np.clip(np.round(weighted), 0, 255).astype(np.uint8)

    return pixel_rgb, ignored_mask


def kmeans_cluster(pixel_rgb: np.ndarray, ignored_mask: np.ndarray,
                   color_num: int, add_coord: bool, lam: float):
    h, w = pixel_rgb.shape[:2]
    valid_positions = np.flatnonzero(~ignored_mask.ravel())
    valid_rgb = pixel_rgb.reshape(-1, 3)[valid_positions] / 255.0

    if valid_rgb.shape[0] == 0:
        raise ValueError("整幅图像在 '透明度>80%' 规则下全部被忽略，没有可用像素。请降低阈值或换图。")

    if add_coord:
        xs = (valid_positions % w) / (w - 1 if w > 1 else 1)
        ys = (valid_positions // w) / (h - 1 if h > 1 else 1)
        coords = np.column_stack((xs, ys)) * lam
        features = np.hstack((valid_rgb, coords))
    else:
        features = valid_rgb

    kmeans = KMeans(n_clusters=min(color_num, features.shape[0]), random_state=0, n_init=10)
    kmeans.fit(features)
    cluster_colors = (kmeans.cluster_centers_[:, :3] * 255).astype(np.uint8)
    return kmeans, cluster_colors, valid_positions


def map_to_palette(cluster_colors: np.ndarray, palettes_json: str, palette_name: str):
    with open(palettes_json, "r", encoding="utf-8") as f:
        palettes = json.load(f)
    if palette_name not in palettes:
        raise ValueError(f"未找到调色板：{palette_name}")
    selected_palette = palettes[palette_name]
    palette_names = [c["name"] for c in selected_palette]
    palette_rgbs = np.array([hex_to_rgb(c["color"]) for c in selected_palette], dtype=np.uint8)

    tree = KDTree(palette_rgbs)
    mapped_center_idx = []
    for c in cluster_colors:
        _, idx = tree.query(c)
        mapped_center_idx.append(int(idx))
    return np.array(mapped_center_idx, dtype=int), palette_rgbs, palette_names


def build_color_field(kmeans, mapped_center_idx, valid_positions, palette_rgbs,
                      out_h: int, out_w: int):
    labels_valid = kmeans.labels_.astype(int)
    mapped_idx_flat = np.full(out_h * out_w, -1, dtype=int)
    mapped_idx_flat[valid_positions] = mapped_center_idx[labels_valid]

    result_rgb = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255
    valid_mask = mapped_idx_flat.reshape(out_h, out_w) >= 0
    result_rgb[valid_mask] = palette_rgbs[mapped_idx_flat[valid_mask.ravel()]].reshape(-1, 3)
    return mapped_idx_flat, result_rgb


def stats_from_field(mapped_idx_flat: np.ndarray, palette_rgbs: np.ndarray, palette_names: list[str]):
    valid_idxs = mapped_idx_flat[mapped_idx_flat >= 0]
    counts = Counter(valid_idxs.tolist())
    stats = []
    for idx, cnt in counts.items():
        name = palette_names[idx]
        rgb = tuple(int(v) for v in palette_rgbs[idx])
        stats.append((name, rgb_to_hex(rgb), rgb, int(cnt)))
    stats.sort(key=lambda x: x[3], reverse=True)
    return stats


def render(canvas_W, canvas_H, width, height, CELL_PX,
           origin_x, origin_y, grid_W, grid_H,
           mapped_idx_flat, palette_rgbs, palette_names, stats,
           draw_grid=True, grid_thickness=1,
           # 字体与布局：
           label_font_path=None, label_font_scale=0.45, label_max_width_ratio=0.7,
           axis_font_scale=0.5, outer_pad=56,
           axis_top_band=None, axis_left_band=None, inner_gap=12,
           legend_conf=None, palette_title="PALETTE", out_image="out.png"):
    """
    渲染整张图（坐标/网格/主格/图例）。
    仅在主函数里计算好尺寸、原点，再调用本函数渲染。
    """
    canvas = Image.new("RGB", (canvas_W, canvas_H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # --- 主格内容 ---
    base_cell_font_px = max(8, int(CELL_PX * label_font_scale))
    for y in range(height):
        for x in range(width):
            idx = mapped_idx_flat[y * width + x]
            x0 = origin_x + x * CELL_PX
            y0 = origin_y + y * CELL_PX
            x1 = x0 + CELL_PX
            y1 = y0 + CELL_PX

            if idx < 0:
                continue

            fill_rgb = tuple(int(v) for v in palette_rgbs[idx])
            name = palette_names[idx]
            draw.rectangle([x0, y0, x1, y1], fill=fill_rgb)

            text_color = pick_text_color(fill_rgb)
            max_w = int(CELL_PX * label_max_width_ratio)
            font = fit_font_to_cell(draw, name, base_cell_font_px, max_w, label_font_path)
            tw, th = text_size(draw, name, font)
            tx = x0 + (CELL_PX - tw) // 2
            ty = y0 + (CELL_PX - th) // 2

            # 轻描边增强对比
            offset = 1
            outline = (255, 255, 255) if text_color == (0, 0, 0) else (0, 0, 0)
            for dx, dy in [(-offset, 0), (offset, 0), (0, -offset), (0, offset)]:
                draw.text((tx + dx, ty + dy), name, font=font, fill=outline)
            draw.text((tx, ty), name, font=font, fill=text_color)

    # --- 网格 ---
    if draw_grid and grid_thickness > 0:
        grid_color = (200, 200, 200)
        for x in range(width + 1):
            xg = origin_x + x * CELL_PX
            draw.line([(xg, origin_y), (xg, origin_y + grid_H)], fill=grid_color, width=grid_thickness)
        for y in range(height + 1):
            yg = origin_y + y * CELL_PX
            draw.line([(origin_x, yg), (origin_x + grid_W, yg)], fill=grid_color, width=grid_thickness)

    # --- 顶部/左侧坐标编号 ---
    axis_font_px = max(8, int(CELL_PX * axis_font_scale))
    axis_font = load_font(axis_font_px, label_font_path)

    # 顶部列号
    for x in range(width):
        label = str(x + 1)
        tw, th = text_size(draw, label, axis_font)
        tx = origin_x + x * CELL_PX + (CELL_PX - tw) // 2
        ty = outer_pad + ((axis_top_band or th + 8) - th) // 2
        draw.text((tx, ty), label, font=axis_font, fill=(0, 0, 0))

    # 左侧行号
    for y in range(height):
        label = str(y + 1)
        tw, th = text_size(draw, label, axis_font)
        tx = outer_pad + (axis_left_band or tw + 12) - tw
        ty = origin_y + y * CELL_PX + (CELL_PX - th) // 2
        draw.text((tx, ty), label, font=axis_font, fill=(0, 0, 0))

    # --- 图例 ---
    if stats:
        lg = legend_conf or {}
        legend_x = origin_x
        legend_y = origin_y + grid_H + lg.get("top_gap", 28)
        title_font = load_font(lg.get("title_size", 18), label_font_path)
        text_font = load_font(lg.get("text_size", 16), label_font_path)

        title = f"Palette Name: {palette_title.upper()}, Totol Color: {len(stats)}"
        draw.text((legend_x, legend_y), title, font=title_font, fill=(0, 0, 0))

        _, legend_title_h = text_size(draw, title, title_font)
        legend_y += legend_title_h + 12

        sw_w = lg.get("swatch_w", 48)
        sw_h = lg.get("swatch_h", 24)
        gap_x = lg.get("gap_x", 24)
        gap_y = lg.get("gap_y", 16)
        cols  = max(1, lg.get("cols", 3))
        text_block_w = lg.get("text_block_w", 280)

        for i, (name, hexv, rgb, cnt) in enumerate(stats):
            col = i % cols
            row = i // cols
            x0 = legend_x + col * (sw_w + gap_x + text_block_w)
            y0 = legend_y + row * (sw_h + gap_y)
            draw.rectangle([x0, y0, x0 + sw_w, y0 + sw_h], fill=rgb, outline=(0, 0, 0))
            text = f"{name}  #{hexv.upper()}  ×{cnt}"
            draw.text((x0 + sw_w + 12, y0 + (sw_h - lg.get('text_size', 16)) // 2),
                      text, fill=(0, 0, 0), font=text_font)

    canvas.save(out_image)
    return canvas


def main():
    # 1) 读取并像素化
    pixel_rgb, ignored_mask = load_and_pixelate(IMG_PATH, OUTPUT_SIZE, ALPHA_IGNORE_THRESHOLD)

    # 2) KMeans 聚类
    kmeans, cluster_colors, valid_positions = kmeans_cluster(
        pixel_rgb, ignored_mask, COLOR_NUM, ADD_COORD, LAMBDA
    )

    # 3) 映射到调色板
    mapped_center_idx, palette_rgbs, palette_names = map_to_palette(
        cluster_colors, PALETTES_JSON, PALETTE_NAME
    )

    # 4) 构造整幅图对应的颜色索引与可视化 RGB
    height = width = OUTPUT_SIZE
    mapped_idx_flat, _ = build_color_field(
        kmeans, mapped_center_idx, valid_positions, palette_rgbs, height, width
    )

    # 5) 用量统计
    stats = stats_from_field(mapped_idx_flat, palette_rgbs, palette_names)

    # 6) 计算画布布局尺寸
    grid_W, grid_H = width * CELL_PX, height * CELL_PX

    # 估算轴带尺寸（用虚拟画笔测量）
    dummy = Image.new("RGB", (10, 10), (255, 255, 255))
    ddraw = ImageDraw.Draw(dummy)
    axis_font_px = max(8, int(CELL_PX * AXIS_FONT_SCALE))
    axis_font = load_font(axis_font_px, LABEL_FONT_PATH)
    max_col_text = str(width) if width >= 10 else "10"
    w_num, h_num = text_size(ddraw, max_col_text, axis_font)
    top_band = AXIS_TOP_BAND if AXIS_TOP_BAND is not None else (h_num + 8)
    left_band = AXIS_LEFT_BAND if AXIS_LEFT_BAND is not None else (w_num + 12)

    # 图例高度估算
    legend_conf = dict(
        swatch_w=LEGEND_SWATCH_W, swatch_h=LEGEND_SWATCH_H,
        gap_x=LEGEND_GAP_X, gap_y=LEGEND_GAP_Y,
        cols=LEGEND_COLS, text_size=LEGEND_TEXT_SIZE,
        title_size=LEGEND_TITLE_SIZE, top_gap=LEGEND_TOP_GAP,
        text_block_w=280
    )

    rows = math.ceil(len(stats) / max(1, LEGEND_COLS))
    legend_title_font = load_font(LEGEND_TITLE_SIZE, LABEL_FONT_PATH)
    _, legend_title_h = text_size(ddraw, f"Palette: {PALETTE_NAME} | Colors: {len(stats)}", legend_title_font)
    legend_h = 0
    if len(stats) > 0:
        legend_h = legend_title_h + 12 + rows * (LEGEND_SWATCH_H + LEGEND_GAP_Y) - (LEGEND_GAP_Y if rows > 0 else 0)

    # 计算最终画布尺寸与原点
    canvas_W = OUTER_PAD + left_band + INNER_GAP + grid_W + OUTER_PAD
    canvas_H = OUTER_PAD + top_band + INNER_GAP + grid_H + (LEGEND_TOP_GAP + legend_h if len(stats) > 0 else 0) + OUTER_PAD
    origin_x = OUTER_PAD + left_band + INNER_GAP
    origin_y = OUTER_PAD + top_band + INNER_GAP

    # 7) 渲染
    render(
        canvas_W, canvas_H, width, height, CELL_PX,
        origin_x, origin_y, grid_W, grid_H,
        mapped_idx_flat, palette_rgbs, palette_names, stats,
        draw_grid=DRAW_GRID, grid_thickness=GRID_THICKNESS,
        label_font_path=LABEL_FONT_PATH, label_font_scale=LABEL_FONT_SCALE,
        label_max_width_ratio=LABEL_MAX_WIDTH_RATIO,
        axis_font_scale=AXIS_FONT_SCALE, outer_pad=OUTER_PAD,
        axis_top_band=top_band, axis_left_band=left_band, inner_gap=INNER_GAP,
        legend_conf=legend_conf, palette_title=PALETTE_NAME, out_image=OUT_IMAGE
    )
    print(f"[OK] 已导出像素成品图: {OUT_IMAGE}")


if __name__ == "__main__":
    main()
