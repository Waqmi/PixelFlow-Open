# PixelFlow v2

PixelFlow 是一个本地优先的桌面工具，用于批量整理电商商品素材并输出主图。它不会生成或修改商品本身；图片处理以等比例缩放、裁切、背景延展和 Logo 叠加为主。

本目录是 PixelFlow 当前唯一的主源码和构建基准。项目根目录的 `02_构建配置/` 只提供兼容转发入口，不再维护另一份实现。

## 功能

- 输出商品白底图、细节图、透明产品图、唯品专享图和模特图。
- 支持上衣、裤装、鞋、帽子、袜子、包和配件等品类比例模板。
- 素材确认页支持商品白底图、细节图和模特图三类；启用本地模型后，模特局部图会自动按细节图规则处理。
- 模特图不使用生成式填充；全身人物会按品类安全聚焦，局部图会铺满画面，避免白边。
- 支持替换方图/竖图、深色/浅色四种 Logo，并可预览实际效果。
- 自动并发数会按设备情况保守选择，兼顾 Windows、Apple Silicon Mac 和旧款 Intel Mac。
- 完整批次会在本机 PixelFlow 缓存目录中短暂保存处理记录：未变化素材复用已有输出，新增或变化素材增量生成；缓存 7 天后自动失效，不会写入或随输出文件夹发送；失败项可导出报告并单独重试。

## 默认资源与本地模型

公开版本附带的默认 Logo 是通用 `YOUR LOGO` 占位资源。请在“设置 → 品牌”中替换为你拥有使用权的 PNG Logo。

本仓库不附带 ONNX 人体姿态模型。若要使用“本地模型辅助判断”，请自行选择合法、兼容 ONNX Runtime 的模型，并在设置中选择其所在文件夹。模型无法加载时，应用会自动回退规则处理。

## 下载与兼容性

`v2.0.0-beta.1` 提供三个版本：

- Windows x64：Windows 10 / 11。
- macOS Apple Silicon：macOS 11 及以上。
- macOS Catalina Intel：已发布的 `v2.0.0-beta.1` 不包含 ONNX Runtime，本地模型功能会回退规则处理。当前源码的 Catalina 构建会内置 `onnxruntime==1.16.3`；在真实 Catalina 设备完成启动和模型推理验收前，不应替换已发布附件。

## 从源码构建

macOS Apple Silicon：

```bash
./build/macOS/build_mac_app.sh
```

从项目根目录运行 `02_构建配置/macOS/build_mac_app.sh` 也可以，脚本会转发到本目录的 v2 构建入口。

Windows x64：在 Windows 10 / 11 上运行：

```bat
build\windows\build_windows.bat
```

从项目根目录运行 `02_构建配置/Windows/build_windows.bat` 也会转发到本目录的 v2 构建入口。

Catalina Intel：在 Intel Mac 上创建 Python 3.9 环境、安装 `build/macOS/requirements-catalina.txt`（含 `onnxruntime==1.16.3`）后运行：

```bash
./build/macOS/build_catalina_app.sh
```

## 可选 API 辅助

API 只用于素材分类和构图建议，最终图片仍由本地引擎处理。请不要把 API Key 写入源码或提交到仓库：

```bash
export OPENAI_API_KEY="你的 API Key"
export PIXELFLOW_OPENAI_MODEL="gpt-5.6"
```

## License

本项目以 [MIT License](LICENSE) 发布。
