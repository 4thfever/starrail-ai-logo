from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .client import generate_image


ROOT_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT_DIR / "assets"
PROMPTS_DIR = ROOT_DIR / "prompts"
OUTPUTS_DIR = ROOT_DIR / "outputs"

ORIGINAL_BG = ASSETS_DIR / "bg.png"
ORIGINAL_TEXT = ASSETS_DIR / "text.png"

BACKGROUND_PROMPT = PROMPTS_DIR / "generate_background.md"
TEXT_PROMPT = PROMPTS_DIR / "generate_text.md"

IMAGE_SIZE = "1536x1024"
TEXT_WIDTH_RATIO = 0.88
DEFAULT_REFERENCE_TEXT = "未提供参考文字；如果有外部参考图，请主要参考外部参考图。"


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


def ensure_outputs_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_prompt(path: Path, reference_text: str) -> str:
    text = reference_text.strip() or DEFAULT_REFERENCE_TEXT
    return _read_prompt(path).replace("[REFERENCE_TEXT]", text)


def reference_images_for_layer(
    spec: LayerSpec,
    reference_image_path: str | None,
) -> list[Path]:
    references = [spec.structure_reference]

    if reference_image_path:
        references.append(Path(reference_image_path))
    return references


def save_bytes(image_bytes: bytes, prefix: str) -> str:
    ensure_outputs_dir()
    output_path = OUTPUTS_DIR / f"{_timestamp()}_{prefix}.png"
    output_path.write_bytes(image_bytes)
    return str(output_path)


def save_image(image: Image.Image, prefix: str) -> str:
    ensure_outputs_dir()
    output_path = OUTPUTS_DIR / f"{_timestamp()}_{prefix}.png"
    image.save(output_path)
    return str(output_path)


def trim_transparent_bounds(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if bbox is None:
        return rgba
    return rgba.crop(bbox)


def prepare_text_layer(path: str) -> str:
    with Image.open(path) as image:
        text_layer = trim_transparent_bounds(image)
    return save_image(text_layer, "text_layer")


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
    return save_bytes(image_bytes, prefix)


async def generate_layer_to_file(
    spec: LayerSpec,
    reference_text: str,
    reference_image_path: str | None,
) -> str:
    return await generate_image_to_file(
        build_prompt(spec.prompt_path, reference_text),
        reference_images_for_layer(spec, reference_image_path),
        prefix=spec.output_prefix,
    )


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
        progress_callback(0.14, desc="并行生成背景层和文字层")
    background_task = generate_layer_to_file(
        background_spec,
        reference_text,
        reference_image_path,
    )
    text_task = generate_layer_to_file(
        text_spec,
        reference_text,
        reference_image_path,
    )
    background_path, raw_text_path = await asyncio.gather(background_task, text_task)

    if progress_callback is not None:
        progress_callback(0.90, desc="准备文字图层")
    text_path = prepare_text_layer(raw_text_path)

    if progress_callback is not None:
        progress_callback(0.95, desc="合成预览图")
    composite_path = compose_paths(background_path, text_path)
    with Image.open(composite_path) as image:
        preview = image.convert("RGBA")
    status = (
        "生成完成。\n"
        f"背景：{background_path}\n"
        f"文字：{text_path}\n"
        f"合成图：{composite_path}"
    )
    return preview, composite_path, status
