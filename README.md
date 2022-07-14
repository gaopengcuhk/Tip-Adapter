# Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification
Official implementation of the **ECCV 2022** paper ['Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification'](https://arxiv.org/abs/2111.03930).
## Introduction
Tip-Adapter is a training-free adaption method for CLIP to conduct few-shot classification, which not only inherits the training-free advantage of zero-shot CLIP but also performs comparably to those training-required approaches. Tip-Adapter constructs the adapter via a key-value cache model from the few-shot training set, and updates the prior knowledge encoded in CLIP by feature retrieval. On top of that, the performance of Tip-Adapter can be further boosted to be state-of-the-art by fine-tuning the cache model for only 10x fewer epochs than existing approaches, which is both effective and efficient.

<div align="center">
  <img src="cache_model.png"/>
</div>

## Requirements
## Get Started

## Contributors
Peng Gao, [Renrui Zhang](https://github.com/ZrrSkywalker)

## Acknowledgement
This repo benefits from [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch) and [CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter). Thanks for their wonderful works.

## Citation

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn and gaopeng@pjlab.org.cn.
