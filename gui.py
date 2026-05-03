from __future__ import annotations

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - shown only when running the app.
    raise SystemExit("Gradio is not installed. Install it with: pip install gradio") from exc

from PIL import Image

from src.img_gen.pipeline import ASSETS_DIR, LOGS_DIR, OUTPUTS_DIR, generate_logo_layers

CSS = """
.hint {
    color: #5f6673;
    font-size: 0.92rem;
    line-height: 1.55;
}
.section-title {
    margin: 0 0 8px;
    font-weight: 700;
    font-size: 1.02rem;
}
.compact textarea {
    min-height: 132px !important;
}
"""


async def generate_layers(
    reference_text: str,
    reference_image_path: str | None,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[Image.Image, str, str]:
    return await generate_logo_layers(reference_text, reference_image_path, progress)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Star Rail AI Logo Generator") as demo:
        gr.Markdown(
            """
            # Star Rail AI Logo Generator

            输入参考文字、上传参考图，或者两者都给。系统会生成背景和透明文字层，
            然后自动合成为一张预览图。
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML('<div class="section-title">1. 输入参考</div>')
                reference_text = gr.Textbox(
                    label="参考文字（可选）",
                    lines=5,
                    elem_classes=["compact"],
                    placeholder="例如：幽蓝火焰、乌鸦、冷月、优雅、墓园、银白与深蓝紫色调",
                )
                reference_image = gr.Image(
                    label="参考图（可选）",
                    type="filepath",
                    sources=["upload", "clipboard"],
                )
                gr.HTML('<div class="section-title">2. 生成</div>')
                generate_button = gr.Button("生成 Logo", variant="primary", size="lg")
                status = gr.Textbox(label="状态", lines=4, interactive=False)

            with gr.Column(scale=5):
                gr.HTML('<div class="section-title">3. 预览</div>')
                preview = gr.Image(
                    label="合成预览",
                    type="pil",
                    buttons=["download", "fullscreen"],
                    interactive=False,
                )
                gr.HTML(
                    """
                    <div class="hint">
                    当前版本先固定居中合成，后续再加入手动拖拽和缩放。
                    </div>
                    """
                )
                download_button = gr.DownloadButton(
                    "下载当前合成图",
                    variant="secondary",
                    size="lg",
                )

        generate_button.click(
            generate_layers,
            inputs=[reference_text, reference_image],
            outputs=[preview, download_button, status],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(css=CSS, allowed_paths=[str(OUTPUTS_DIR), str(LOGS_DIR), str(ASSETS_DIR)])
