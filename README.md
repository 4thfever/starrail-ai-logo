# Star Rail AI Logo

这是一个用于生成《崩坏：星穹铁道》风格化 logo 的小工具。

项目目标是：输入参考文字、参考图，或两者同时输入，自动生成一组新的 logo 背景层和透明文字层，并自动叠加成一张可下载的合成图。

## 使用方式

1. 配置图像生成 API：

```powershell
copy src\img_gen\image_api.env.example src\img_gen\image_api.env
```

然后在 `src/img_gen/image_api.env` 中填写 `IMAGE_API_KEY`。

2. 安装依赖：

```powershell
pip install gradio pillow openai
```

3. 启动界面：

```powershell
python gui.py
```

## 基本流程

1. 输入参考文字或上传参考图。
2. 点击 `生成 Logo`。
3. 查看合成预览图。
4. 点击 `下载当前合成图`。

生成结果会保存在 `outputs/` 目录中。
