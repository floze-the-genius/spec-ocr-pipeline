# Spec OCR Pipeline

[English](README.md) | [Русский](README.ru.md) | [简体中文](README.zh-CN.md)

[![CI](https://github.com/floze-the-genius/spec-ocr-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/floze-the-genius/spec-ocr-pipeline/actions/workflows/ci.yml)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Spec OCR Pipeline 是一个跨平台命令行工具，可将工程规格书 PDF 转换为可检查的规范化数据行
和可编辑的 Excel 工作簿。对于带有可用文本层的 PDF，它提供完全本地、确定性的处理流程；
对于扫描件，可以使用系统安装的 Tesseract；对于复杂版式，还可以显式选择本地或外部 AI
提供商。

> **公开 Alpha 版本：**核心流程已经可用并通过测试，但提供商配置、命令参数和输出细节仍可能
> 发生变化。OCR 和 AI 结果均应视为草稿；在实际使用前，请务必检查数量、单位、行边界、合计
> 和工作簿公式。

![Spec OCR Pipeline 演示：将合成 PDF 转换为结构化数据行和 Excel 工作簿](docs/assets/demo.jpg)

## 输出内容

- 便于检查、比较和二次处理的规范化 JSONL；
- 可编辑的单工作表 XLSX 文件；
- 可选的、基于模板的商业报价工作簿；
- OCR 数据、质量报告、性能报告和运行元数据；
- 明确的提供商选择，可识别数据是在本地处理还是发送到外部服务。

仓库内置的演示文件完全由合成数据生成，可安全用于日志、截图和公开问题。真实文档版式各不
相同，命令成功结束并不代表每个识别结果都正确。

## 快速开始

从 GitHub 的 **Code** 菜单复制 HTTPS 或 SSH 地址，然后克隆仓库：

```bash
git clone https://github.com/floze-the-genius/spec-ocr-pipeline.git
cd spec-ocr-pipeline
```

创建并激活虚拟环境。

Linux 或 macOS：

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

安装带 OCR 支持的用户环境：

```bash
python -m pip install -e ".[ocr]"
```

贡献者应改为安装开发环境：

```bash
python -m pip install -e ".[dev]"
```

在仓库根目录运行单命令合成演示：

```bash
spec-ocr process examples/demo/specification.pdf --output-dir demo-output --quote
```

该演示使用 PDF 文本层，**不会**调用外部 AI 服务。它会生成：

| 文件 | 用途 |
| --- | --- |
| `demo-output/ocr.json` | 提取出的页面文本、词元、元数据和提供商信息 |
| `demo-output/rows.jsonl` | 可检查的规范化规格数据行 |
| `demo-output/result.xlsx` | 可编辑的单工作表 Excel 文件 |
| `demo-output/quote.xlsx` | 基于模板生成的商业报价工作簿 |
| `demo-output/run.json` | 版本、输入、参数、输出文件名和外部 AI 标志 |

除非显式添加 `--force`，命令不会覆盖已有输出。安全的单命令流程支持具有可识别表格文本层的
PDF；扫描件或复杂文档需要按照[使用指南](docs/usage.md)显式选择提供商。

## 平台支持

| 功能 | Linux | macOS | Windows |
| --- | --- | --- | --- |
| 核心文本型 PDF / Excel 流程 | 原生支持 | 原生支持 | 原生支持 |
| Tesseract OCR | 系统可执行文件 | 系统可执行文件 | 系统可执行文件 |
| 本地 AI | x86_64 + NVIDIA | Apple Silicon + MLX | x64 + WSL2 + NVIDIA |

需要 Python 3.11–3.14。Tesseract 可执行文件和语言包必须通过操作系统安装，本仓库不会捆绑
它们。目前不声明支持 Intel Mac 本地 AI、Windows ARM 或 Windows 原生 CUDA。

平台安装说明：

- [Linux 安装](docs/installation/linux.md)
- [macOS 安装](docs/installation/macos.md)
- [Windows 安装](docs/installation/windows.md)

## 支持的 OCR 和 AI 模型

OCR 提取和语义行映射是两个独立阶段。核心文本型 PDF 流程不需要 AI 模型。

| 阶段 | 支持的选项 | 支持环境 |
| --- | --- | --- |
| PDF 提取 | `pdf-text`，无需模型 | Linux、macOS、Windows；本地处理 |
| 传统 OCR | Tesseract 和用户安装的语言数据 | Linux、macOS、Windows；本地处理 |
| 视觉文档 OCR | PaddleOCR-VL | macOS 或 Windows WSL2；使用本地模型文件 |
| 实验性视觉 OCR | 来自已审计本地快照的 dots.mocr | Apple Silicon MLX 或 Linux/WSL2 Transformers 运行时 |
| MLX 映射 | Qwen3 8B/14B、Qwen3.6 35B A3B 4-bit | Apple Silicon macOS；根据模型需要 16–64 GB 统一内存 |
| GGUF 映射 | Qwen3.6 27B：`qwen36-safe`、`qwen36-balanced`、`qwen36-quality` | Linux 或 Windows WSL2；llama.cpp；32–48 GB RAM |
| CUDA 映射 | Qwen3 8B/14B、Qwen3-VL 8B、实验性 Qwen3.6 27B GPTQ | Linux 或 Windows WSL2 + NVIDIA；12–24 GB VRAM |
| API 映射 | 用户选择的 OpenAI 兼容模型或 OpenCode 模型 | 本地回环服务器或经过明确批准的外部提供商 |

运行以下命令查看当前机器上的托管本地模型目录：

```bash
spec-ocr models status --json
```

完整的模型 aliases、上游 ID、后端、磁盘空间、RAM/VRAM 要求和安全边界请参阅
[提供商和支持的模型](docs/providers.md)。模型不会捆绑在仓库中；清单中的条目也不代表已经
确认其许可证或硬件兼容性。

## 提供商和隐私

核心 PDF 文本层流程和系统 Tesseract 均在本地运行。可选的语义映射提供商可以是本地服务，
也可以是外部服务。外部 AI 提供商可能接收 OCR 文本；只有在显式启用时才会接收页面图像。
处理真实文档前，请阅读[提供商](docs/providers.md)和[隐私说明](docs/privacy.md)。

AI 映射默认采用 fail-closed 策略：如果不存在受支持的确定性表格层，`map-ocr` 要求显式指定
`--ai-client`。请只使用自己的提供商账户，不要提交或分享 API 密钥、OpenCode/浏览器会话、
刷新令牌、Cookie、`.env` 文件或导出的身份验证状态。公开 issue 中不得附加真实文档或由其
生成的 OCR 和工作簿数据。

## 文档

- 安装：[Linux](docs/installation/linux.md)、[macOS](docs/installation/macos.md)、
  [Windows](docs/installation/windows.md)
- [使用指南](docs/usage.md)
- [OCR、AI 提供商和支持的模型](docs/providers.md)
- [隐私和凭据](docs/privacy.md)
- [架构](docs/architecture.md)
- [故障排除](docs/troubleshooting.md)
- [发布流程](docs/releasing.md)
- [贡献指南](CONTRIBUTING.md)
- [安全策略](SECURITY.md)
- [支持](SUPPORT.md)
- [变更日志](CHANGELOG.md)
- [MIT 许可证](LICENSE)

## 开发

安装 `.[dev]` 后运行：

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest
python scripts/check_public_tree.py
```

CI 会在 Linux、macOS 和 Windows 上检查受支持的 Python 版本，并验证构建后的软件包。
依赖特定硬件的 OCR 和 AI 运行时仍应在对应机器上实际测试；请在 pull request 中如实说明未验证
的组合。

欢迎提交带有合成样例、回归测试和明确平台结果的针对性改进。创建 pull request 前请阅读
[CONTRIBUTING.md](CONTRIBUTING.md)；安全漏洞请按照 [SECURITY.md](SECURITY.md) 中的私密流程
报告。

## 许可证

Spec OCR Pipeline 使用 [MIT 许可证](LICENSE)。
