from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - shown only when running the app.
    raise SystemExit(
        "Gradio is not installed. Install it with: pip install gradio"
    ) from exc

from PIL import Image, ImageStat

from src.img_gen import generate_image


ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
PROMPTS_DIR = ROOT_DIR / "prompts"
OUTPUTS_DIR = ROOT_DIR / "outputs"

ORIGINAL_LOGO = ASSETS_DIR / "original.png"
ORIGINAL_BG = ASSETS_DIR / "bg.png"
ORIGINAL_TEXT = ASSETS_DIR / "text.png"

PROMPT_FILES = {
    ("参考文字", "background"): PROMPTS_DIR / "ref_text_to_background.md",
    ("参考文字", "text"): PROMPTS_DIR / "ref_text_to_text.md",
    ("参考图", "background"): PROMPTS_DIR / "ref_image_to_background.md",
    ("参考图", "text"): PROMPTS_DIR / "ref_image_to_text.md",
}

DEFAULT_CENTER_X = 50.0
DEFAULT_CENTER_Y = 50.0
DEFAULT_TEXT_WIDTH_PCT = 84.0
DEFAULT_IMAGE_SIZE = "1536x1024"


def _ensure_outputs_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _build_prompt(
    *,
    mode: str,
    layer: str,
    reference_text: str,
    reference_image_path: str | None,
) -> str:
    prompt = _read_prompt(PROMPT_FILES[(mode, layer)])
    prompt = prompt.replace("[REFERENCE_TEXT]", reference_text.strip())
    prompt = prompt.replace("[REFERENCE_IMAGE]", reference_image_path or "uploaded reference image")
    return prompt


def _reference_images_for(mode: str, reference_image_path: str | None, layer: str) -> list[Path]:
    if layer == "background":
        references = [ORIGINAL_LOGO, ORIGINAL_BG, ORIGINAL_TEXT]
    else:
        references = [ORIGINAL_LOGO, ORIGINAL_TEXT, ORIGINAL_BG]

    if mode == "参考图":
        if not reference_image_path:
            raise ValueError("请选择参考图，或切换到“参考文字”模式。")
        references.append(Path(reference_image_path))

    return references


def _save_generated_image(image_bytes: bytes, prefix: str) -> str:
    _ensure_outputs_dir()
    output_path = OUTPUTS_DIR / f"{_timestamp()}_{prefix}.png"
    output_path.write_bytes(image_bytes)
    return str(output_path)


def _image_has_useful_alpha(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False
    alpha = image.getchannel("A")
    extrema = alpha.getextrema()
    return extrema[0] < 250


def _border_pixels(image: Image.Image, *, inset: int = 8) -> list[tuple[int, int, int]]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    inset = max(1, min(inset, width // 6 or 1, height // 6 or 1))

    pixels: list[tuple[int, int, int]] = []
    for x in range(width):
        for y in range(inset):
            pixels.append(rgb.getpixel((x, y)))
            pixels.append(rgb.getpixel((x, height - 1 - y)))
    for y in range(inset, height - inset):
        for x in range(inset):
            pixels.append(rgb.getpixel((x, y)))
            pixels.append(rgb.getpixel((width - 1 - x, y)))
    return pixels


def _estimated_background_color(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    border = _border_pixels(rgb)
    if not border:
        stat = ImageStat.Stat(rgb)
        return tuple(int(channel) for channel in stat.mean[:3])

    channels = list(zip(*border))
    return tuple(int(sorted(channel)[len(channel) // 2]) for channel in channels)


def _distance_from_color(pixel: tuple[int, int, int], color: tuple[int, int, int]) -> float:
    return (
        abs(pixel[0] - color[0])
        + abs(pixel[1] - color[1])
        + abs(pixel[2] - color[2])
    ) / 3


def _auto_transparency_threshold(image: Image.Image, background: tuple[int, int, int]) -> float:
    distances = sorted(_distance_from_color(pixel, background) for pixel in _border_pixels(image))
    if not distances:
        return 18.0

    p90 = distances[int(len(distances) * 0.90)]
    # Keep the cutoff conservative so dark text on a dark preview matte is not erased.
    return min(34.0, max(10.0, p90 + 7.0))


def _with_estimated_transparency(image: Image.Image) -> Image.Image:
    """Fallback for text layers returned with a flat opaque background."""
    if _image_has_useful_alpha(image):
        return image.convert("RGBA")

    rgba = image.convert("RGBA")
    background = _estimated_background_color(rgba)
    threshold = _auto_transparency_threshold(rgba, background)
    converted: list[tuple[int, int, int, int]] = []

    for red, green, blue, alpha in rgba.getdata():
        distance = _distance_from_color((red, green, blue), background)
        if distance <= threshold:
            converted.append((red, green, blue, 0))
            continue

        if distance >= threshold + 18.0:
            converted.append((red, green, blue, alpha))
            continue

        soft_alpha = min(255, int((distance - threshold) * 14.0))
        converted.append((red, green, blue, min(alpha, max(32, soft_alpha))))

    rgba.putdata(converted)
    return rgba


def _trim_transparent_bounds(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if bbox is None:
        return rgba
    return rgba.crop(bbox)


def _alpha_composite_clipped(canvas: Image.Image, overlay: Image.Image, left: int, top: int) -> None:
    canvas_width, canvas_height = canvas.size
    overlay_width, overlay_height = overlay.size

    crop_left = max(0, -left)
    crop_top = max(0, -top)
    crop_right = min(overlay_width, canvas_width - left)
    crop_bottom = min(overlay_height, canvas_height - top)

    if crop_left >= crop_right or crop_top >= crop_bottom:
        return

    visible_overlay = overlay.crop((crop_left, crop_top, crop_right, crop_bottom))
    canvas.alpha_composite(visible_overlay, (max(0, left), max(0, top)))


def _open_background(path: str | None) -> Image.Image | None:
    if not path:
        return None
    return Image.open(path).convert("RGBA")


def _open_text(path: str | None, auto_transparent: bool) -> Image.Image | None:
    if not path:
        return None
    image = Image.open(path)
    if auto_transparent:
        return _with_estimated_transparency(image)
    return image.convert("RGBA")


def compose_logo(
    background_path: str | None,
    text_path: str | None,
    center_x_pct: float,
    center_y_pct: float,
    text_width_pct: float,
    auto_transparent: bool,
) -> Image.Image | None:
    background = _open_background(background_path)
    text = _open_text(text_path, auto_transparent)

    if background is None and text is None:
        return None
    if background is None:
        return text
    if text is None:
        return background

    canvas = background.copy()
    text = _trim_transparent_bounds(text)
    if text.width <= 0 or text.height <= 0:
        return canvas

    target_width = max(1, int(canvas.width * (text_width_pct / 100.0)))
    target_height = max(1, int(text.height * (target_width / text.width)))
    text = text.resize((target_width, target_height), Image.Resampling.LANCZOS)

    center_x = int(canvas.width * (center_x_pct / 100.0))
    center_y = int(canvas.height * (center_y_pct / 100.0))
    left = center_x - text.width // 2
    top = center_y - text.height // 2
    _alpha_composite_clipped(canvas, text, left, top)
    return canvas


def _save_composite(image: Image.Image | None) -> str | None:
    if image is None:
        return None

    _ensure_outputs_dir()
    output_path = OUTPUTS_DIR / f"{_timestamp()}_composite.png"
    image.save(output_path)
    return str(output_path)


def load_original_layers() -> tuple[str, str, Image.Image | None, float, float, float, str]:
    preview = compose_logo(
        str(ORIGINAL_BG),
        str(ORIGINAL_TEXT),
        DEFAULT_CENTER_X,
        DEFAULT_CENTER_Y,
        DEFAULT_TEXT_WIDTH_PCT,
        True,
    )
    return (
        str(ORIGINAL_BG),
        str(ORIGINAL_TEXT),
        preview,
        DEFAULT_CENTER_X,
        DEFAULT_CENTER_Y,
        DEFAULT_TEXT_WIDTH_PCT,
        "已载入原始背景和原始文字层，可直接测试位置与缩放控制。",
    )


def generate_layers(
    mode: str,
    reference_text: str,
    reference_image_path: str | None,
    image_size: str,
    center_x_pct: float,
    center_y_pct: float,
    text_width_pct: float,
    auto_transparent: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str, Image.Image | None, str]:
    size = None if image_size == "使用配置默认" else image_size

    if mode == "参考文字" and not reference_text.strip():
        raise ValueError("请输入参考文字，或切换到“参考图”模式。")

    progress(0.05, desc="准备 prompt 和参考图")
    background_prompt = _build_prompt(
        mode=mode,
        layer="background",
        reference_text=reference_text,
        reference_image_path=reference_image_path,
    )
    text_prompt = _build_prompt(
        mode=mode,
        layer="text",
        reference_text=reference_text,
        reference_image_path=reference_image_path,
    )

    progress(0.15, desc="生成背景层")
    background_bytes = generate_image(
        background_prompt,
        _reference_images_for(mode, reference_image_path, "background"),
        size=size,
    )
    background_path = _save_generated_image(background_bytes, "background")

    progress(0.62, desc="生成透明文字层")
    text_bytes = generate_image(
        text_prompt,
        _reference_images_for(mode, reference_image_path, "text"),
        size=size,
    )
    text_path = _save_generated_image(text_bytes, "text")

    progress(0.92, desc="合成预览")
    preview = compose_logo(
        background_path,
        text_path,
        center_x_pct,
        center_y_pct,
        text_width_pct,
        auto_transparent,
    )
    return background_path, text_path, preview, f"生成完成：\n背景：{background_path}\n文字：{text_path}"


def generate_background_layer(
    mode: str,
    reference_text: str,
    reference_image_path: str | None,
    image_size: str,
    text_path: str | None,
    center_x_pct: float,
    center_y_pct: float,
    text_width_pct: float,
    auto_transparent: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, Image.Image | None, str]:
    size = None if image_size == "使用配置默认" else image_size
    if mode == "参考文字" and not reference_text.strip():
        raise ValueError("请输入参考文字，或切换到“参考图”模式。")

    progress(0.10, desc="准备背景 prompt 和参考图")
    prompt = _build_prompt(
        mode=mode,
        layer="background",
        reference_text=reference_text,
        reference_image_path=reference_image_path,
    )
    progress(0.25, desc="生成背景层")
    image_bytes = generate_image(
        prompt,
        _reference_images_for(mode, reference_image_path, "background"),
        size=size,
    )
    background_path = _save_generated_image(image_bytes, "background")
    preview = compose_logo(
        background_path,
        text_path,
        center_x_pct,
        center_y_pct,
        text_width_pct,
        auto_transparent,
    )
    return background_path, preview, f"背景生成完成：{background_path}"


def generate_text_layer(
    mode: str,
    reference_text: str,
    reference_image_path: str | None,
    image_size: str,
    background_path: str | None,
    center_x_pct: float,
    center_y_pct: float,
    text_width_pct: float,
    auto_transparent: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, Image.Image | None, str]:
    size = None if image_size == "使用配置默认" else image_size
    if mode == "参考文字" and not reference_text.strip():
        raise ValueError("请输入参考文字，或切换到“参考图”模式。")

    progress(0.10, desc="准备文字 prompt 和参考图")
    prompt = _build_prompt(
        mode=mode,
        layer="text",
        reference_text=reference_text,
        reference_image_path=reference_image_path,
    )
    progress(0.25, desc="生成透明文字层")
    image_bytes = generate_image(
        prompt,
        _reference_images_for(mode, reference_image_path, "text"),
        size=size,
    )
    text_path = _save_generated_image(image_bytes, "text")
    preview = compose_logo(
        background_path,
        text_path,
        center_x_pct,
        center_y_pct,
        text_width_pct,
        auto_transparent,
    )
    return text_path, preview, f"文字生成完成：{text_path}"


def update_composite(
    background_path: str | None,
    text_path: str | None,
    center_x_pct: float,
    center_y_pct: float,
    text_width_pct: float,
    auto_transparent: bool,
) -> Image.Image | None:
    return compose_logo(
        background_path,
        text_path,
        center_x_pct,
        center_y_pct,
        text_width_pct,
        auto_transparent,
    )


def export_composite(
    background_path: str | None,
    text_path: str | None,
    center_x_pct: float,
    center_y_pct: float,
    text_width_pct: float,
    auto_transparent: bool,
) -> tuple[str | None, str]:
    image = compose_logo(
        background_path,
        text_path,
        center_x_pct,
        center_y_pct,
        text_width_pct,
        auto_transparent,
    )
    saved_path = _save_composite(image)
    if not saved_path:
        return None, "没有可导出的合成图。请先生成或载入图层。"
    return saved_path, f"已导出合成图：{saved_path}"


def nudge_position(
    center_x_pct: float,
    center_y_pct: float,
    text_width_pct: float,
    direction: str,
) -> tuple[float, float, float]:
    step = 2.0
    scale_step = 4.0

    if direction == "left":
        center_x_pct -= step
    elif direction == "right":
        center_x_pct += step
    elif direction == "up":
        center_y_pct -= step
    elif direction == "down":
        center_y_pct += step
    elif direction == "smaller":
        text_width_pct -= scale_step
    elif direction == "larger":
        text_width_pct += scale_step

    return (
        min(120.0, max(-20.0, center_x_pct)),
        min(120.0, max(-20.0, center_y_pct)),
        min(160.0, max(10.0, text_width_pct)),
    )


def handle_preview_select(
    current_x_pct: float,
    current_y_pct: float,
    preview_image: Image.Image | None,
    evt: gr.SelectData,
) -> tuple[float, float]:
    if not evt or evt.index is None:
        return current_x_pct, current_y_pct

    if not isinstance(evt.index, (list, tuple)) or len(evt.index) < 2:
        return current_x_pct, current_y_pct

    x, y = evt.index[0], evt.index[1]
    if preview_image is None:
        return current_x_pct, current_y_pct

    image_width, image_height = preview_image.size
    if not image_width or not image_height:
        return current_x_pct, current_y_pct

    return (x / image_width) * 100.0, (y / image_height) * 100.0


def build_app() -> gr.Blocks:
    css = """
    .hint {
        color: #5f6673;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    """

    with gr.Blocks(title="Star Rail AI Logo Generator", css=css) as demo:
        gr.Markdown(
            """
            # Star Rail AI Logo Generator

            先根据参考文字或参考图分别生成背景层与透明文字层，再把文字叠到背景上。
            如果 AI 生成的文字位置或大小不够协调，可以用右侧的中心点、缩放和微调按钮重新排版，不需要重新生成。
            """
        )

        background_state = gr.State(value=None)
        text_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=4):
                mode = gr.Radio(
                    choices=["参考文字", "参考图"],
                    value="参考文字",
                    label="参考类型",
                )
                reference_text = gr.Textbox(
                    label="参考文字",
                    lines=5,
                    placeholder="例如：幽蓝火焰、乌鸦、冷月、优雅、墓园、银白与深蓝紫色调",
                )
                reference_image = gr.Image(
                    label="参考图",
                    type="filepath",
                    sources=["upload", "clipboard"],
                )
                image_size = gr.Dropdown(
                    label="生成尺寸",
                    choices=["1536x1024", "使用配置默认", "1024x1024"],
                    value=DEFAULT_IMAGE_SIZE,
                )
                auto_transparent = gr.Checkbox(
                    label="文字层若不是透明底，自动尝试抠除纯色背景",
                    value=True,
                )

                with gr.Row():
                    generate_button = gr.Button("生成背景 + 文字", variant="primary")
                    load_original_button = gr.Button("载入原始素材预览")
                with gr.Row():
                    generate_background_button = gr.Button("只生成背景")
                    generate_text_button = gr.Button("只生成文字")

                status = gr.Textbox(label="状态", lines=4, interactive=False)

            with gr.Column(scale=5):
                preview = gr.Image(
                    label="合成预览",
                    type="pil",
                    interactive=True,
                    show_download_button=True,
                )
                gr.HTML(
                    """
                    <div class="hint">
                    调整方式：拖动滑杆做精确控制；按钮用于小步微调；若当前 Gradio 版本支持图片点击事件，
                    点击预览图可把文字中心移动到点击位置。
                    </div>
                    """
                )

        with gr.Row():
            with gr.Column(scale=3):
                center_x = gr.Slider(
                    -20,
                    120,
                    value=DEFAULT_CENTER_X,
                    step=0.5,
                    label="文字中心 X（背景宽度百分比）",
                )
                center_y = gr.Slider(
                    -20,
                    120,
                    value=DEFAULT_CENTER_Y,
                    step=0.5,
                    label="文字中心 Y（背景高度百分比）",
                )
                text_width = gr.Slider(
                    10,
                    160,
                    value=DEFAULT_TEXT_WIDTH_PCT,
                    step=0.5,
                    label="文字宽度（占背景宽度百分比）",
                )

            with gr.Column(scale=2):
                with gr.Row():
                    up_button = gr.Button("上移")
                with gr.Row():
                    left_button = gr.Button("左移")
                    right_button = gr.Button("右移")
                with gr.Row():
                    down_button = gr.Button("下移")
                with gr.Row():
                    smaller_button = gr.Button("缩小")
                    larger_button = gr.Button("放大")

                export_button = gr.Button("导出当前合成图", variant="secondary")
                exported_file = gr.File(label="导出的合成图")

        generation_inputs = [
            mode,
            reference_text,
            reference_image,
            image_size,
            center_x,
            center_y,
            text_width,
            auto_transparent,
        ]
        generation_outputs = [background_state, text_state, preview, status]

        generate_button.click(
            generate_layers,
            inputs=generation_inputs,
            outputs=generation_outputs,
        )

        generate_background_button.click(
            generate_background_layer,
            inputs=[
                mode,
                reference_text,
                reference_image,
                image_size,
                text_state,
                center_x,
                center_y,
                text_width,
                auto_transparent,
            ],
            outputs=[background_state, preview, status],
        )

        generate_text_button.click(
            generate_text_layer,
            inputs=[
                mode,
                reference_text,
                reference_image,
                image_size,
                background_state,
                center_x,
                center_y,
                text_width,
                auto_transparent,
            ],
            outputs=[text_state, preview, status],
        )

        load_original_button.click(
            load_original_layers,
            inputs=[],
            outputs=[
                background_state,
                text_state,
                preview,
                center_x,
                center_y,
                text_width,
                status,
            ],
        )

        compose_inputs = [
            background_state,
            text_state,
            center_x,
            center_y,
            text_width,
            auto_transparent,
        ]
        for control in [center_x, center_y, text_width, auto_transparent]:
            control.change(update_composite, inputs=compose_inputs, outputs=preview)

        nudge_buttons: list[tuple[Any, str]] = [
            (left_button, "left"),
            (right_button, "right"),
            (up_button, "up"),
            (down_button, "down"),
            (smaller_button, "smaller"),
            (larger_button, "larger"),
        ]
        for button, direction in nudge_buttons:
            button.click(
                lambda x, y, scale, d=direction: nudge_position(x, y, scale, d),
                inputs=[center_x, center_y, text_width],
                outputs=[center_x, center_y, text_width],
            ).then(update_composite, inputs=compose_inputs, outputs=preview)

        preview.select(
            handle_preview_select,
            inputs=[center_x, center_y, preview],
            outputs=[center_x, center_y],
        ).then(update_composite, inputs=compose_inputs, outputs=preview)

        export_button.click(
            export_composite,
            inputs=compose_inputs,
            outputs=[exported_file, status],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
