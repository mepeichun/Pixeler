# utils.py
from PIL import ImageFont

# -------- 颜色/文本相关 --------
def hex_to_rgb(hex_str: str):
    hex_str = hex_str.strip().lower().lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "{:02x}{:02x}{:02x}".format(*rgb)

def luminance(rgb):
    r, g, b = [v/255.0 for v in rgb]
    return 0.2126*r + 0.7152*g + 0.0722*b

def pick_text_color(bg_rgb, threshold: float = 0.6):
    """根据背景亮度选择黑/白文字，threshold 越大越容易选黑字"""
    return (0, 0, 0) if luminance(bg_rgb) > threshold else (255, 255, 255)

# -------- 字体/文本测量 --------
def load_font(size: int, font_path: str | None = None):
    """安全加载字体：有路径用路径，失败或无路径用默认字体"""
    try:
        return ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

def text_size(draw, text: str, font):
    """兼容 Pillow 新旧版本的文字尺寸测量"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def fit_font_to_cell(draw, text: str, font_base_size: int, max_w: int,
                     font_path: str | None = None, min_size: int = 6):
    """
    将文字缩放到不超过 max_w 的宽度，返回合适字体对象。
    """
    size = font_base_size
    while size > min_size:
        font = load_font(size, font_path)
        w, _ = text_size(draw, text, font)
        if w <= max_w:
            return font
        size -= 1
    return load_font(min_size, font_path)
