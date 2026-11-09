<div align="center">

# UniHIR

### Draft, Verify, Restore：基于统一多模态大模型的自迭代历史铭文修复

[**English**](README.md) | [**简体中文**](README_CN.md)

[![ACL 2026 Main](https://img.shields.io/badge/ACL_2026-Main-8A2BE2)](https://aclanthology.org/2026.acl-long.1254/)
[![论文](https://img.shields.io/badge/Paper-ACL_Anthology-B31B1B)](https://aclanthology.org/2026.acl-long.1254/)
[![模型](https://img.shields.io/badge/Model-ModelScope-624AFF)](https://www.modelscope.cn/models/zzxfzyy/UniHIR)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-green)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

**张宇一<sup>&#42;</sup> · 刘俊乐<sup>&#42;</sup> · 张沛荣<sup>&#42;</sup> · 刘建良 · 杨振华 · 金连文<sup>†</sup>**

华南理工大学 · 深度学习与视觉计算实验室

<sup>&#42;</sup> 共同一作 &nbsp;&nbsp; <sup>†</sup> 通讯作者

</div>

> **UniHIR 是首个面向端到端历史铭文修复的统一多模态大模型。** 它在同一框架中依次完成损伤位置与缺失内容草拟、自校验与迭代优化，以及整页外观修复。

## 🔥 最新动态

- **2026-07-05** — 论文发表于 **ACL 2026 Main**。[[论文页面](https://aclanthology.org/2026.acl-long.1254/)] [[PDF](https://aclanthology.org/2026.acl-long.1254.pdf)]
- **2026-07-05** — 推理代码与 UniHIR 预训练模型正式发布。

## ✨ 方法概览

<p align="center"><img src="fig/unihir_pipeline.png" alt="UniHIR 统一历史铭文修复流程" width="100%"></p>

UniHIR 将传统的任务分离式流水线统一为一个模型，并完成三个相互衔接的阶段：

1. **草稿引导定位（Draft-Guided Localization）**：联合定位损伤区域并预测难以辨认的文字。
2. **层级自迭代（Hierarchical Self-Refinement）**：从宏观和微观两个层面反复修正漏检、误检、位置偏差与语义不一致结果。
3. **外观修复（Appearance Restoration）**：将优化后的内容重新渲染到文档中，并保持整页字体和风格一致。

### 核心贡献

- 提出面向端到端历史铭文修复的统一多模态大模型框架。
- 设计草稿引导定位与层级自迭代机制，实现迭代理解与自我校正。
- 提出 **UHIRFactory**，支持高分辨率输入和长序列条件下的分阶段显存高效指令微调。
- 构建覆盖损伤定位、层级优化与外观修复的 **HIRBench**。
- 在文字修复准确率和视觉质量上取得优异表现，同时保持整页一致性。

## 📦 项目资源

| 资源 | 链接 | 状态 |
|---|---|---|
| 论文 | [ACL Anthology](https://aclanthology.org/2026.acl-long.1254/) · [PDF](https://aclanthology.org/2026.acl-long.1254.pdf) | 已发布 |
| UniHIR 预训练模型 | [ModelScope](https://www.modelscope.cn/models/zzxfzyy/UniHIR) | 已发布 |
| 推理代码 | [本仓库](https://github.com/ZZXF11/UniHIR) | 已发布 |
| HIRBench | [申请入口](http://121.41.49.212:9000/) | 可申请获取 |

## 🚀 快速开始

### 1. 克隆仓库并创建环境

```bash
git clone https://github.com/ZZXF11/UniHIR.git
cd UniHIR
conda create -n unihir python=3.10 -y
conda activate unihir
pip install -r requirements.txt
```

### 2. 下载预训练模型

从 [ModelScope](https://www.modelscope.cn/models/zzxfzyy/UniHIR) 下载模型权重，将模型目录放在 `./UniHIR`，或通过 `--model_path` 指定实际路径。

### 3. 执行推理

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --img_path examples/FS_12_159_2.jpg \
  --ref_count 6 \
  --save_path ./results \
  --model_path ./UniHIR
```

命令会在 `./results` 中生成修复结果图，以及包含中间步骤和合并后修复预测的文本文件。

> [!NOTE]
> 推荐使用具有 **80 GB 显存**的 NVIDIA A100 级别 GPU。实际显存需求会随图像分辨率和自迭代次数变化。

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--img_path` | `examples/FS_12_159_2.jpg` | 待修复的历史铭文图像路径 |
| `--ref_count` | `6` | 层级自迭代次数 |
| `--save_path` | `./results` | 修复图像和文本预测保存目录 |
| `--model_path` | `./UniHIR` | UniHIR 预训练模型本地路径 |

## 🧩 HIRBench

<p align="center"><img src="fig/hirbench.png" alt="HIRBench 任务设计" width="100%"></p>

HIRBench 包含三个子集，分别为草稿引导定位（HIRBench-DGL）、层级自迭代（HIRBench-HSR）和外观修复（HIRBench-AR）。

### 数据申请与负责任使用

原始数据来自互联网等公开渠道，相关版权归原始提供方所有。整理和标注后的 HIRBench 数据集仅限非商业科研用途，目前仅向高校及科研机构授权。

- 请通过实验室在线平台提交本人签署的申请表。申请人须为高校或科研机构的全职工作人员。为方便审核，建议加盖单位公章，二级部门公章亦可。👉 [点击申请](http://121.41.49.212:9000/)
- 提交申请时，请列出或附上近 6 年内的 1–2 篇论文，以说明您或您的团队从事 OCR、历史文档分析与修复、文档图像处理等相关研究。
- 申请通过审核后，我们将提供数据集下载链接。
- 所有用户均须遵守数据集使用条件，否则授权将被撤销。

<details>
<summary><strong>数据目录与样本格式</strong></summary>

数据集提供原始目录和 ZIP 压缩包两种下载形式。任选一种下载，保留文件名与目录结构。若下载压缩包，按下表选择文件，统一解压到 `HIRBench/`，并单独下载 `dataset_info.json`。

| 子集 | 标注压缩包 | 图片压缩包 |
|---|---|---|
| DGL | `HIRBench-DGL.zip` | `images-DGL.zip` |
| HSR | `HIRBench-HSR.zip` | `images-HSR-targets.zip` 与 `images-HSR-inputs.zip.001` 至 `.006` 全部六卷 |
| AR | `HIRBench-AR.zip` | `images-AR.zip.001` 至 `.003` 全部三卷 |

同组分卷放在同一目录，从 `.001` 开始解压。解压后的结构如下。

```text
HIRBench/
├── dataset_info.json
├── HIRBench-DGL/{real,syn}/         # JSON 标注
├── HIRBench-HSR/                   # JSON 标注
├── HIRBench-AR/{real,syn,inscription_synthesis}/
└── images/
    ├── DGL/{real,syn}/{inputs,targets}/
    ├── HSR/{inputs,targets}/
    ├── AR/{real,syn}/{inputs,drafts,targets}/
    └── AR/inscription_synthesis/{drafts,targets}/
```

大括号表示多个独立子目录。每个 UTF-8 JSON 文件对应一条样本。`conversations` 保存对话消息，每条含 `from`（`human` 为指令，`gpt` 为监督回答）与 `value`（文本内容）。`images` 按任务顺序列出图片路径。**图片路径相对于该 JSON 文件所在目录**，例如 `sample_path.parent / sample["images"][0]`。

| 子集 | `images` 中的图片顺序 |
|---|---|
| DGL | 输入页面 → 定位草稿目标图 |
| HSR | 含噪检测草稿 → 精修草稿目标图 |
| AR（`real`、`syn`） | 输入页面 → 字符草稿 → 修复目标图 |
| AR（`inscription_synthesis`） | 字符草稿 → 重复引用同一草稿 → 修复目标图 |

各任务的最后一张图均为监督目标。DGL 和 HSR 的框坐标归一化至 0–1000，格式为 `<|box_start|>x1,y1,x2,y2,character<|box_end|>`。

</details>

## 📊 实验结果

### 定量比较

<p align="center"><img src="fig/eval.png" alt="UniHIR 定量实验结果" width="100%"></p>

### 定性比较

<p align="center"><img src="fig/vis1.png" alt="UniHIR 定性修复结果" width="100%"></p>

## 🗺️ 开源计划

- [x] 发布推理代码
- [x] 发布 UniHIR 预训练模型
- [x] 发布 HIRBench

## 💙 致谢

[BAGEL](https://github.com/bytedance-seed/BAGEL) · [AutoHDR](https://github.com/SCUT-DLVCLab/AutoHDR) · [DiffHDR](https://github.com/yeungchenwa/HDR) · [HisDoc1B](https://github.com/SCUT-DLVCLab/HisDoc1B) · [MegaHan97K](https://github.com/SCUT-DLVCLab/MegaHan97K)

## 📜 许可与版权

代码与数据以 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) 许可提供，仅限非商业科研使用。商业使用请联系金连文教授：`eelwjin@scut.edu.cn`。

版权所有 © 2026 华南理工大学[深度学习与视觉计算实验室](http://www.dlvc-lab.net)。

## ✒️ 引用

```bibtex
@inproceedings{zhang-etal-2026-draft,
  title     = {Draft, Verify, Restore: Self-Refining Historical Inscription Restoration with a Unified MLLM},
  author    = {Zhang, Yuyi and Liu, Junle and Zhang, Peirong and Liu, Jianliang and Yang, Zhenhua and Jin, Lianwen},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = jul,
  year      = {2026},
  address   = {San Diego, California, United States},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2026.acl-long.1254/},
  doi       = {10.18653/v1/2026.acl-long.1254},
  pages     = {27216--27231}
}
```
