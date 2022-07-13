# Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification
Official implementation of the **ECCV 2022** paper ['Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification'](https://arxiv.org/abs/2111.03930).
## Introduction
Tip-Adapter provides faster convergence and better performance than CLIP-Adapter by initializing the adapter with a cache model.

<div align="center">
  <img src="cache_model.png"/>
</div>

## Implementation
Put ``tip_adapter_ImageNet.py`` into clip's folder and run 

    python tip_adapter_ImageNet.py

you will get 65.51% on ImageNet validation set.

This repo will be completed in a few days.


## Contributors
Peng Gao, Renrui Zhang

## Acknowledgment
CLIP, CoOp and CLIP-Adapter
