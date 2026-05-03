from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from src.img_gen.pipeline import OUTPUTS_DIR, generate_logo_layers


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "inputs"
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass
class BatchResult:
    input: str
    ok: bool
    output: str | None = None
    status: str | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量读取 inputs/ 下的参考图，逐张生成 Star Rail 风格 logo。",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="参考图目录，默认是项目根目录下的 inputs/。",
    )
    parser.add_argument(
        "--reference-text",
        default="",
        help="可选参考文字；会和每张参考图一起送入生成流程。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出将处理的图片，不调用图像生成 API。",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="任意一张图片失败后立即停止；默认会继续处理后续图片。",
    )
    return parser.parse_args()


def find_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {input_dir}")

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_progress_callback(index: int, total: int, image_path: Path):
    label = f"[{index}/{total}] {image_path.name}"

    def progress(value: float, desc: str = "") -> None:
        percent = max(0, min(100, round(value * 100)))
        if desc:
            print(f"{label} {percent:>3}% - {desc}", flush=True)

    return progress


def write_manifest(results: list[BatchResult]) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = OUTPUTS_DIR / f"{timestamp}_batch_manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def output_path_for_input(image_path: Path) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR / f"{image_path.stem}.png"


def move_composite_to_named_output(output_path: str, image_path: Path, status: str) -> tuple[str, str]:
    generated_path = Path(output_path)
    named_path = output_path_for_input(image_path)

    if generated_path.resolve() != named_path.resolve():
        generated_path.replace(named_path)
        status = status.replace(str(generated_path), str(named_path))

    return str(named_path), status


async def generate_one(
    image_path: Path,
    *,
    index: int,
    total: int,
    reference_text: str,
) -> BatchResult:
    started = perf_counter()
    try:
        preview, output_path, status = await generate_logo_layers(
            reference_text,
            str(image_path),
            make_progress_callback(index, total, image_path),
        )
        preview.close()
        output_path, status = move_composite_to_named_output(
            output_path,
            image_path,
            status,
        )
        elapsed = perf_counter() - started
        print(
            f"[{index}/{total}] 完成: {image_path.name} -> {output_path} "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        return BatchResult(
            input=str(image_path),
            ok=True,
            output=output_path,
            status=status,
            elapsed_seconds=round(elapsed, 2),
        )
    except Exception as exc:
        elapsed = perf_counter() - started
        print(
            f"[{index}/{total}] 失败: {image_path.name} - {exc} "
            f"({elapsed:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
        return BatchResult(
            input=str(image_path),
            ok=False,
            error=str(exc),
            elapsed_seconds=round(elapsed, 2),
        )


async def run_batch(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.expanduser().resolve()
    images = find_images(input_dir)

    if not images:
        print(f"没有在 {input_dir} 找到可处理图片。")
        return 0

    print(f"找到 {len(images)} 张图片:")
    for image_path in images:
        print(f"- {image_path}")

    if args.dry_run:
        print("dry-run 模式结束，没有调用图像生成 API。")
        return 0

    results: list[BatchResult] = []
    total = len(images)
    for index, image_path in enumerate(images, start=1):
        result = await generate_one(
            image_path,
            index=index,
            total=total,
            reference_text=args.reference_text,
        )
        results.append(result)

        if not result.ok and args.stop_on_error:
            break

    manifest_path = write_manifest(results)
    success_count = sum(1 for result in results if result.ok)
    fail_count = len(results) - success_count
    print(
        f"批处理结束: 成功 {success_count}，失败 {fail_count}，"
        f"manifest: {manifest_path}"
    )

    return 1 if fail_count else 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_batch(args))


if __name__ == "__main__":
    raise SystemExit(main())
