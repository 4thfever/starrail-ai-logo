from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter

from .client import generate_image


ROOT_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT_DIR / "assets"
PROMPTS_DIR = ROOT_DIR / "prompts"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = ROOT_DIR / "logs"

ORIGINAL_BG = ASSETS_DIR / "bg.png"
ORIGINAL_TEXT = ASSETS_DIR / "text.png"

BACKGROUND_PROMPT = PROMPTS_DIR / "generate_background.md"
TEXT_PROMPT = PROMPTS_DIR / "generate_text.md"

IMAGE_SIZE = "1536x1024"
TEXT_WIDTH_RATIO = 0.88
DEFAULT_REFERENCE_TEXT = "未提供参考文字；如果有外部参考图，请主要参考外部参考图。"
KEY_COLORS = (
    ("#00FF00", (0, 255, 0)),
    ("#FF00FF", (255, 0, 255)),
)
KEY_BACKGROUND_MIN_RATIO = 0.82
KEY_BACKGROUND_DISTANCE = 70
KEY_ALPHA_HARD_DISTANCE = 55
KEY_ALPHA_SOFT_DISTANCE = 210
KEY_SPILL_DISTANCE = 250
KEY_SPILL_STRENGTH = 0.9


@dataclass(frozen=True)
class LayerSpec:
    prompt_path: Path
    structure_reference: Path
    output_prefix: str


LAYER_SPECS = {
    "background": LayerSpec(
        prompt_path=BACKGROUND_PROMPT,
        structure_reference=ORIGINAL_BG,
        output_prefix="background",
    ),
    "text": LayerSpec(
        prompt_path=TEXT_PROMPT,
        structure_reference=ORIGINAL_TEXT,
        output_prefix="text_raw",
    ),
}


class KeyBackgroundError(RuntimeError):
    pass


def ensure_outputs_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_logs_dir() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_prompt(
    path: Path,
    reference_text: str,
    key_color: tuple[str, tuple[int, int, int]] | None = None,
) -> str:
    text = reference_text.strip() or DEFAULT_REFERENCE_TEXT
    prompt = _read_prompt(path).replace("[REFERENCE_TEXT]", text)
    if key_color is None:
        return prompt

    key_hex, key_rgb = key_color
    return (
        prompt.replace("[KEY_COLOR]", key_hex).replace(
            "[KEY_RGB]",
            f"{key_rgb[0]}, {key_rgb[1]}, {key_rgb[2]}",
        )
    )


def reference_images_for_layer(
    spec: LayerSpec,
    reference_image_path: str | None,
) -> list[Path]:
    references = [spec.structure_reference]

    if reference_image_path:
        references.append(Path(reference_image_path))
    return references


def save_bytes(image_bytes: bytes, prefix: str, *, directory: Path = OUTPUTS_DIR) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{_timestamp()}_{prefix}.png"
    output_path.write_bytes(image_bytes)
    return str(output_path)


def save_image(image: Image.Image, prefix: str, *, directory: Path = OUTPUTS_DIR) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{_timestamp()}_{prefix}.png"
    image.save(output_path)
    return str(output_path)


def trim_transparent_bounds(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if bbox is None:
        return rgba
    return rgba.crop(bbox)


def _color_distance_sq(
    color: tuple[int, int, int],
    key_rgb: tuple[int, int, int],
) -> int:
    return sum((channel - key) ** 2 for channel, key in zip(color, key_rgb))


def _edge_pixels(
    image: Image.Image,
    band_ratio: float = 0.05,
) -> list[tuple[int, int, int]]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    band_x = max(1, int(width * band_ratio))
    band_y = max(1, int(height * band_ratio))
    regions = (
        rgb.crop((0, 0, width, band_y)),
        rgb.crop((0, height - band_y, width, height)),
        rgb.crop((0, 0, band_x, height)),
        rgb.crop((width - band_x, 0, width, height)),
    )
    pixels: list[tuple[int, int, int]] = []
    for region in regions:
        pixels.extend(region.getdata())
    return pixels


def key_background_ratio(
    image: Image.Image,
    key_rgb: tuple[int, int, int],
) -> float:
    threshold_sq = KEY_BACKGROUND_DISTANCE**2
    pixels = _edge_pixels(image)
    if not pixels:
        return 0.0
    matching = sum(
        1 for pixel in pixels if _color_distance_sq(pixel, key_rgb) <= threshold_sq
    )
    return matching / len(pixels)


def is_key_background(image: Image.Image, key_rgb: tuple[int, int, int]) -> bool:
    return key_background_ratio(image, key_rgb) >= KEY_BACKGROUND_MIN_RATIO


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _remove_key_spill(
    red: int,
    green: int,
    blue: int,
    key_rgb: tuple[int, int, int],
) -> tuple[int, int, int]:
    channels = [red, green, blue]
    key_channels = [index for index, value in enumerate(key_rgb) if value >= 250]
    non_key_channels = [index for index in range(3) if index not in key_channels]
    if not key_channels or not non_key_channels:
        return red, green, blue

    spill_distance = (_color_distance_sq((red, green, blue), key_rgb)) ** 0.5
    if spill_distance >= KEY_SPILL_DISTANCE:
        return red, green, blue

    spill_factor = KEY_SPILL_STRENGTH * _smoothstep(
        (KEY_SPILL_DISTANCE - spill_distance) / KEY_SPILL_DISTANCE
    )
    non_key_max = max(channels[index] for index in non_key_channels)
    for index in key_channels:
        allowed = non_key_max
        if channels[index] > allowed:
            channels[index] = int(round(channels[index] * (1 - spill_factor) + allowed * spill_factor))

    return channels[0], channels[1], channels[2]


def remove_key_background(
    image: Image.Image,
    key_rgb: tuple[int, int, int],
) -> Image.Image:
    rgba = image.convert("RGBA")
    hard_sq = KEY_ALPHA_HARD_DISTANCE**2
    soft = KEY_ALPHA_SOFT_DISTANCE
    soft_range = max(1, soft - KEY_ALPHA_HARD_DISTANCE)
    output: list[tuple[int, int, int, int]] = []

    for red, green, blue, original_alpha in rgba.getdata():
        distance_sq = _color_distance_sq((red, green, blue), key_rgb)
        if distance_sq <= hard_sq:
            output.append((0, 0, 0, 0))
            continue

        distance = distance_sq**0.5
        alpha_factor = _smoothstep((distance - KEY_ALPHA_HARD_DISTANCE) / soft_range)
        alpha = int(round(original_alpha * alpha_factor))
        if alpha <= 0:
            output.append((0, 0, 0, 0))
            continue

        if alpha < 255:
            normalized_alpha = alpha / 255
            red = int(
                round((red - key_rgb[0] * (1 - normalized_alpha)) / normalized_alpha)
            )
            green = int(
                round((green - key_rgb[1] * (1 - normalized_alpha)) / normalized_alpha)
            )
            blue = int(
                round((blue - key_rgb[2] * (1 - normalized_alpha)) / normalized_alpha)
            )
            red = max(0, min(255, red))
            green = max(0, min(255, green))
            blue = max(0, min(255, blue))

        red, green, blue = _remove_key_spill(red, green, blue, key_rgb)
        output.append((red, green, blue, alpha))

    cutout = Image.new("RGBA", rgba.size)
    cutout.putdata(output)
    alpha = cutout.getchannel("A").filter(ImageFilter.GaussianBlur(radius=0.35))
    cutout.putalpha(alpha)
    return cutout


def prepare_text_layer(path: str, key_rgb: tuple[int, int, int]) -> str:
    with Image.open(path) as image:
        if not is_key_background(image, key_rgb):
            ratio = key_background_ratio(image, key_rgb)
            raise KeyBackgroundError(
                f"文字层边缘 Key 色匹配度过低：{ratio:.1%}"
            )
        text_layer = trim_transparent_bounds(remove_key_background(image, key_rgb))
    return save_image(text_layer, "text_layer", directory=LOGS_DIR)


def alpha_composite_centered(background: Image.Image, text: Image.Image) -> Image.Image:
    canvas = background.convert("RGBA")
    text_layer = trim_transparent_bounds(text)
    target_width = int(canvas.width * TEXT_WIDTH_RATIO)
    target_height = int(text_layer.height * (target_width / text_layer.width))
    text_layer = text_layer.resize((target_width, target_height), Image.Resampling.LANCZOS)

    left = (canvas.width - text_layer.width) // 2
    top = (canvas.height - text_layer.height) // 2
    canvas.alpha_composite(text_layer, (left, top))
    return canvas


def compose_paths(background_path: str, text_path: str) -> str:
    with Image.open(background_path) as background, Image.open(text_path) as text:
        composite = alpha_composite_centered(background, text)
    return save_image(composite, "composite")


async def generate_image_to_file(
    prompt: str,
    references: list[Path],
    *,
    prefix: str,
) -> str:
    image_bytes = await asyncio.to_thread(
        generate_image,
        prompt,
        references,
        size=IMAGE_SIZE,
    )
    return save_bytes(image_bytes, prefix, directory=LOGS_DIR)


async def generate_layer_to_file(
    spec: LayerSpec,
    reference_text: str,
    reference_image_path: str | None,
    key_color: tuple[str, tuple[int, int, int]] | None = None,
) -> str:
    return await generate_image_to_file(
        build_prompt(spec.prompt_path, reference_text, key_color),
        reference_images_for_layer(spec, reference_image_path),
        prefix=spec.output_prefix,
    )


async def generate_text_layer_with_key(
    spec: LayerSpec,
    reference_text: str,
    reference_image_path: str | None,
    key_color: tuple[str, tuple[int, int, int]],
) -> tuple[str, str, str, tuple[int, int, int]]:
    raw_text_path = await generate_layer_to_file(
        spec,
        reference_text,
        reference_image_path,
        key_color,
    )
    key_hex, key_rgb = key_color
    text_path = prepare_text_layer(raw_text_path, key_rgb)
    return raw_text_path, text_path, key_hex, key_rgb


async def generate_logo_layers(
    reference_text: str,
    reference_image_path: str | None,
    progress_callback: Callable[..., object] | None = None,
) -> tuple[Image.Image, str, str]:
    if not reference_text.strip() and not reference_image_path:
        raise ValueError("请至少输入参考文字或上传参考图。")

    if progress_callback is not None:
        progress_callback(0.06, desc="准备背景 prompt")
    background_spec = LAYER_SPECS["background"]
    text_spec = LAYER_SPECS["text"]

    if progress_callback is not None:
        progress_callback(0.14, desc="并行生成背景层和绿幕 Key 文字层")
    background_task = asyncio.create_task(
        generate_layer_to_file(
            background_spec,
            reference_text,
            reference_image_path,
        )
    )

    key_error: KeyBackgroundError | None = None
    try:
        raw_text_path, text_path, key_hex, key_rgb = await generate_text_layer_with_key(
            text_spec,
            reference_text,
            reference_image_path,
            KEY_COLORS[0],
        )
    except KeyBackgroundError as exc:
        key_error = exc
        if progress_callback is not None:
            progress_callback(0.72, desc="绿幕背景不合格，改用洋红 Key 重试")
        raw_text_path, text_path, key_hex, key_rgb = await generate_text_layer_with_key(
            text_spec,
            reference_text,
            reference_image_path,
            KEY_COLORS[1],
        )

    background_path = await background_task

    if progress_callback is not None:
        progress_callback(0.95, desc="合成预览图")
    composite_path = compose_paths(background_path, text_path)
    with Image.open(composite_path) as image:
        preview = image.convert("RGBA")
    status = (
        "生成完成。\n"
        f"背景：{background_path}\n"
        f"文字原图：{raw_text_path}\n"
        f"文字：{text_path}\n"
        f"抠图 Key 色：{key_hex} RGB{key_rgb}\n"
        + (f"首次 Key 检查：{key_error}\n" if key_error is not None else "")
        + f"合成图：{composite_path}"
    )
    return preview, composite_path, status
