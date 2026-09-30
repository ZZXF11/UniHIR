<div align="center">

# UniHIR

### Draft, Verify, Restore: Self-Refining Historical Inscription Restoration with a Unified MLLM

[**English**](README.md) | [**简体中文**](README_CN.md)

[![ACL 2026 Main](https://img.shields.io/badge/ACL_2026-Main-8A2BE2)](https://aclanthology.org/2026.acl-long.1254/)
[![Paper](https://img.shields.io/badge/Paper-ACL_Anthology-B31B1B)](https://aclanthology.org/2026.acl-long.1254/)
[![Model](https://img.shields.io/badge/Model-ModelScope-624AFF)](https://www.modelscope.cn/models/zzxfzyy/UniHIR)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-green)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![GitHub stars](https://img.shields.io/github/stars/ZZXF11/UniHIR?style=social)](https://github.com/ZZXF11/UniHIR)

**Yuyi Zhang<sup>*</sup> · Junle Liu<sup>*</sup> · Peirong Zhang<sup>*</sup> · Jianliang Liu · Zhenhua Yang · Lianwen Jin<sup>†</sup>**

South China University of Technology · Deep Learning and Vision Computing Lab

<sup>*</sup> Equal contribution &nbsp;&nbsp; <sup>†</sup> Corresponding author

</div>

> **UniHIR is the first unified multimodal large language model for end-to-end historical inscription restoration.** It drafts damage locations and illegible content, verifies and refines its own predictions, and restores the full-page appearance in one unified framework.

## 🔥 News

- **2026-07-05** — The paper is published at **ACL 2026 Main**. [[Paper](https://aclanthology.org/2026.acl-long.1254/)] [[PDF](https://aclanthology.org/2026.acl-long.1254.pdf)]
- **2026-07-05** — Inference code and pretrained UniHIR weights are released.

## ✨ Overview

<p align="center">
  <img src="fig/unihir_pipeline.png" alt="UniHIR unified historical inscription restoration pipeline" width="100%">
</p>

UniHIR replaces task-separated restoration pipelines with a single model that performs three connected stages:

1. **Draft-Guided Localization** — jointly localizes damaged regions and predicts illegible characters.
2. **Hierarchical Self-Refinement** — repeatedly corrects missing, redundant, misaligned, or semantically inconsistent predictions at both macro and micro levels.
3. **Appearance Restoration** — renders the refined content back into the document while preserving page-level typography and style.

### Key contributions

- A unified MLLM formulation for end-to-end Historical Inscription Restoration (HIR).
- Draft-Guided Localization and Hierarchical Self-Refinement for iterative reasoning and self-correction.
- **UHIRFactory**, a step-wise, memory-efficient instruction-tuning strategy for high-resolution inputs and long sequences.
- **HIRBench**, covering damage localization, hierarchical refinement, and appearance restoration.
- Strong restoration accuracy and visual quality without sacrificing full-page consistency.

## 📦 Resources

| Resource | Link | Status |
|---|---|---|
| Paper | [ACL Anthology](https://aclanthology.org/2026.acl-long.1254/) · [PDF](https://aclanthology.org/2026.acl-long.1254.pdf) | Available |
| Pretrained UniHIR | [ModelScope](https://www.modelscope.cn/models/zzxfzyy/UniHIR) | Available |
| Inference code | [This repository](https://github.com/ZZXF11/UniHIR) | Available |
| HIRBench | [Application portal](http://121.41.49.212:9000/) | Release in progress |

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ZZXF11/UniHIR.git
cd UniHIR
```

### 2. Create the environment

```bash
conda create -n unihir python=3.10 -y
conda activate unihir
pip install -r requirements.txt
```

### 3. Download the pretrained model

Download the weights from [ModelScope](https://www.modelscope.cn/models/zzxfzyy/UniHIR) and place the model directory at `./UniHIR`, or provide its location through `--model_path`.

### 4. Run inference

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --img_path examples/FS_12_159_2.jpg \
  --ref_count 6 \
  --save_path ./results \
  --model_path ./UniHIR
```

The command saves two files in `./results`:

- the restored image, using the original input filename;
- a text file containing the intermediate and merged restoration predictions.

> [!NOTE]
> We recommend an NVIDIA A100-class GPU with **80 GB VRAM** for inference. Memory requirements may vary with image resolution and the number of refinement iterations.

### Command-line arguments

| Argument | Default | Description |
|---|---:|---|
| `--img_path` | `examples/FS_12_159_2.jpg` | Input damaged historical inscription image |
| `--ref_count` | `6` | Number of hierarchical refinement iterations |
| `--save_path` | `./results` | Directory for restored images and text predictions |
| `--model_path` | `./UniHIR` | Local path to the pretrained UniHIR model |

## 🧩 HIRBench

<p align="center">
  <img src="fig/hirbench.png" alt="HIRBench task design" width="100%">
</p>

| Subset | Task | Release status |
|---|---|---|
| HIRBench-DGL | Draft-Guided Localization | Coming soon |
| HIRBench-HSR | Hierarchical Self-Refinement | Coming soon |
| HIRBench-AR | Appearance Restoration | Coming soon |

HIRBench is available only for non-commercial research. University and research-institute applicants may submit a request through the [application portal](http://121.41.49.212:9000/). Approved applicants will receive the archive password and must follow all stated usage conditions.

## 📊 Results

### Quantitative comparison

<p align="center">
  <img src="fig/eval.png" alt="UniHIR quantitative evaluation" width="100%">
</p>

### Qualitative comparison

<p align="center">
  <img src="fig/vis1.png" alt="UniHIR qualitative restoration comparison" width="100%">
</p>

## 🗺️ Roadmap

- [x] Release inference code
- [x] Release pretrained UniHIR model
- [ ] Release HIRBench
- [ ] Add lower-memory inference guidance

## ⚖️ Dataset Access and Responsible Use

The original source materials are collected from public channels, including the Internet, and their copyrights remain with the original providers. The curated and annotated dataset is intended solely for non-commercial research and is currently licensed to universities and research institutions.

Applicants must be full-time employees of an eligible institution and must submit a signed application form. An official institutional seal is recommended to facilitate review; a secondary-level department seal is acceptable.

## 💙 Acknowledgements

- [BAGEL](https://github.com/bytedance-seed/BAGEL)
- [AutoHDR](https://github.com/SCUT-DLVCLab/AutoHDR)
- [DiffHDR](https://github.com/yeungchenwa/HDR)
- [HisDoc1B](https://github.com/SCUT-DLVCLab/HisDoc1B)
- [MegaHan97K](https://github.com/SCUT-DLVCLab/MegaHan97K)

## 📜 License and Copyright

The code and dataset are provided under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) for non-commercial research. For commercial use, please contact Prof. Lianwen Jin at `eelwjin@scut.edu.cn`.

Copyright © 2026 [Deep Learning and Vision Computing Lab](http://www.dlvc-lab.net), South China University of Technology.

## ✒️ Citation

If UniHIR is useful in your research, please cite the ACL 2026 paper:

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

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ZZXF11/UniHIR&type=Timeline)](https://star-history.com/#ZZXF11/UniHIR&Timeline)
