<div align=center>

# Draft, Verify, Restore: Self-Refining Historical Inscription Restoration with a Unified MLLM

</div>

<div align=center>

[![Paper](https://img.shields.io/badge/Paper-ACL2026-ff6b6b)](https://aclanthology.org/2026.acl-long.1254/)
[![GitHub ZZXF11](https://img.shields.io/badge/GitHub-ZZXF11-blueviolet?logo=github)](https://github.com/ZZXF11)
[![SCUT DLVC Lab](https://img.shields.io/badge/SCUT-DLVC_Lab-327FE6?logo=Academia&logoColor=white)](http://dlvc-lab.net/lianwen/)
[![Code](https://img.shields.io/badge/Code-UniHIR-yellow)](https://github.com/ZZXF11/UniHIR)

</div>

## Important Note
The original data of the dataset is sourced from public channels such as the Internet, and its copyright shall remain with the original providers. The collated and annotated dataset presented in this case is for non-commercial use only and is currently licensed to universities and research institutions. To apply for the use of this dataset, please fill in the corresponding application form in accordance with the requirements specified on the dataset’s official website. The applicant must be a full-time employee of a university or research institute and is required to sign the application form. For the convenience of review, it is recommended to affix an official seal (a seal of a secondary-level department is acceptable).

## 🌟 Highlights
- **UniHIR**
![Vis_1](fig/unihir_pipeline.png)
- **HIRBench**
![Vis_2](fig/hirbench.png)

- We propose **UniHIR**, a pioneering Unified MLLM for end-to-end Historical Inscription Restoration (HIR), offering a new perspective on HIR.
- UniHIR incorporates two novel designs, Draft-Guided Localization and Hierarchical Self-Refinement, to support iterative reasoning and self-correction for HIR.
- We propose UHIRFactory and construct **HIRBench** to enable step-wise, memory efficient training of UniHIR.
- Extensive experiments demonstrate that our method achieves superior restored-text accuracy and generation quality.

## 📏 Evaluation Result
![Vis_3](fig/eval.png)

## 🌄 Visualization
![vis_4](fig/vis1.png)

## 📅 News
- **2026.07.05**: 🎉🎉 Our [paper](https://aclanthology.org/2026.acl-long.1254/) is accepted by ACL Main.


## 🚧 TODO List

- [x] Release inference code
- [ ] Release pretrained model
- [ ] Release dataset

## 🚧 Installation
Clone this repo:
```bash
git clone https://github.com/ZZXF11/UniHIR
cd UniHIR
```

**Step 0**: Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1**: Create a conda environment and activate it.
```bash
conda create -n unihir python=3.10 -y
conda activate unihir
```

**Step 2**: Install the required packages.
```bash
pip install -r requirements.txt
```

## 📺 Inference

**Step 1**: Download the pretrained model from [ModelScope](https://www.modelscope.cn/models/zzxfzyy/UniHIR) or [Hugging Face](https://huggingface.co/ZZXF11/UniHIR).

**Step 2**: Run inference:
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python infer.py \
    --img_path examples/FS_12_159_2.jpg \
    --ref_count 6 \
    --save_path ./results \
    --model_path ./UniHIR
```

> **Note**: We recommend using a GPU with 80GB VRAM (e.g., NVIDIA A100) for inference.

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--img_path` | str | `examples/FS_12_159_2.jpg` | Path to input damaged historical document image |
| `--ref_count` | int | `6` | Number of refinement iterations |
| `--save_path` | str | `./results` | Directory to save inference results |
| `--model_path` | str | `./UniHIR` | Path to the pretrained UniHIR model directory |



## 🔥 HIRBench
| Dataset | Description | Link | Status |
|---------|-------------|------|--------|
| HIRBench-DGL | Draft-Guided Localization | [Download](#) | 🚧 Coming soon |
| HIRBench-HSR | Hierarchical Self-Refinement | [Download](#) | 🚧 Coming soon |
| HIRBench-AR | Appearance Restoration | [Download](#) | 🚧 Coming soon |


**Note:**
- The HIRBench can only be used for non-commercial research purposes. Scholars or organizations who wish to use the HIRBench can apply through our online platform:  👉 [Apply Here](http://121.41.49.212:9000/)
- We will give you the decompression password after your application has been received and approved.
- All users must follow all use conditions; otherwise, the authorization will be revoked.

## 💙 Acknowledgement
- [BAGEL](https://github.com/bytedance-seed/BAGEL)
- [AutoHDR](https://github.com/SCUT-DLVCLab/AutoHDR)
- [DiffHDR](https://github.com/yeungchenwa/HDR)
- [HisDoc1B](https://github.com/SCUT-DLVCLab/HisDoc1B)
- [MegaHan97K](https://github.com/SCUT-DLVCLab/MegaHan97K)


## 📜 License
The code and dataset should be used and distributed under [(CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/) for non-commercial research purposes.

## ⛔️ Copyright
- This repository can only be used for non-commercial research purposes.
- For commercial use, please contact Prof. Lianwen Jin (eelwjin@scut.edu.cn).
- Copyright 2026, [Deep Learning and Vision Computing Lab (DLVC-Lab)](http://www.dlvc-lab.net), South China University of Technology. 

## ✒️Citation
If you find PosterVerse helpful, please consider giving this repo a ⭐ and citing:
```latex
@article{zhang2026unihir,
      title={Draft, Verify, Restore: Self-Refining Historical Inscription Restoration with a Unified MLLM}, 
      author={Yuyi Zhang, Junle Liu, Peirong Zhang, Jianliang Liu, Zhenhua Yang, Lianwen Jin},
      journal={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
      year={2026},
}
```
Thanks for your support!

## ⭐ Star Rising
[![Star Rising](https://api.star-history.com/svg?repos=ZZXF11/UniHIR&type=Timeline)](https://star-history.com/#ZZXF11/UniHIR&Timeline)
