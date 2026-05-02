from __future__ import annotations

import base64
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .config import ImageGenConfig, load_config

T = TypeVar("T")
RETRY_STATUS_CODES = {408, 409, 429}
ReferenceImages = str | Path | list[str | Path] | tuple[str | Path, ...] | None


def create_client(config: ImageGenConfig | None = None) -> OpenAI:
    resolved = config or load_config()
    return OpenAI(base_url=resolved.base_url, api_key=resolved.api_key)


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRY_STATUS_CODES or exc.status_code >= 500
    return False


def _with_retries(
    operation: Callable[[], T],
    *,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> T:
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_retries or not _is_retryable_error(exc):
                raise
            delay = retry_delay * (2**attempt)
            print(
                f"图像 API 请求失败，{delay:.1f}s 后重试 "
                f"({attempt + 1}/{max_retries}): {exc}"
            )
            time.sleep(delay)
            attempt += 1


def _extract_first_image_b64(response: object) -> str:
    if not getattr(response, "data", None):
        raise RuntimeError("未在响应中找到图片数据")

    first = response.data[0]
    image_b64 = getattr(first, "b64_json", None) or getattr(first, "b64", None)
    if not image_b64:
        raise RuntimeError("未在响应中找到图片 base64 数据")
    return image_b64


def _decode_image_response(response: object) -> bytes:
    return base64.b64decode(_extract_first_image_b64(response))


def _normalize_reference_images(
    reference_images: ReferenceImages,
) -> list[Path]:
    if reference_images is None:
        return []

    if isinstance(reference_images, (str, Path)):
        candidates = [reference_images]
    else:
        candidates = list(reference_images)

    normalized = [Path(image).expanduser() for image in candidates]
    missing = [str(path) for path in normalized if not path.exists()]
    if missing:
        raise FileNotFoundError(f"参考图不存在: {', '.join(missing)}")

    return normalized


def _request_image_edit(
    image_paths: list[Path],
    prompt: str,
    *,
    config: ImageGenConfig | None = None,
    model: str | None = None,
    size: str | None = None,
    n: int = 1,
    collapse_single_image: bool = False,
) -> object:
    resolved = config or load_config()
    client = create_client(resolved)

    with ExitStack() as stack:
        opened_files = [stack.enter_context(path.open("rb")) for path in image_paths]
        image_input = (
            opened_files[0]
            if collapse_single_image and len(opened_files) == 1
            else opened_files
        )
        return client.images.edit(
            model=model or resolved.model,
            image=image_input,
            prompt=prompt,
            n=n,
            size=size or resolved.size,
            response_format="b64_json",
        )


def generate_image_b64(
    prompt: str,
    *,
    config: ImageGenConfig | None = None,
    model: str | None = None,
    size: str | None = None,
    n: int = 1,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    resolved = config or load_config()
    client = create_client(resolved)
    response = _with_retries(
        lambda: client.images.generate(
            model=model or resolved.model,
            prompt=prompt,
            n=n,
            size=size or resolved.size,
            response_format="b64_json",
        ),
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    return _extract_first_image_b64(response)


def generate_image_bytes(
    prompt: str,
    *,
    config: ImageGenConfig | None = None,
    model: str | None = None,
    size: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> bytes:
    image_b64 = generate_image_b64(
        prompt,
        config=config,
        model=model,
        size=size,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    return base64.b64decode(image_b64)


def edit_image_b64(
    image_path: str | Path,
    prompt: str,
    *,
    config: ImageGenConfig | None = None,
    model: str | None = None,
    size: str | None = None,
    n: int = 1,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    response = _with_retries(
        lambda: _request_image_edit(
            [Path(image_path).expanduser()],
            prompt,
            config=config,
            model=model,
            size=size,
            n=n,
            collapse_single_image=True,
        ),
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    return _extract_first_image_b64(response)


def edit_image_bytes(
    image_path: str | Path,
    prompt: str,
    *,
    config: ImageGenConfig | None = None,
    model: str | None = None,
    size: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> bytes:
    image_b64 = edit_image_b64(
        image_path,
        prompt,
        config=config,
        model=model,
        size=size,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    return base64.b64decode(image_b64)


def generate_image(
    prompt: str,
    reference_images: ReferenceImages = None,
    *,
    config: ImageGenConfig | None = None,
    model: str | None = None,
    size: str | None = None,
    n: int = 1,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> bytes:
    references = _normalize_reference_images(reference_images)
    if not references:
        return generate_image_bytes(
            prompt,
            config=config,
            model=model,
            size=size,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    response = _with_retries(
        lambda: _request_image_edit(
            references,
            prompt,
            config=config,
            model=model,
            size=size,
            n=n,
        ),
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    return _decode_image_response(response)
