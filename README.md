Vision Transformers (ViTs) have achieved impressive results in large-scale image classification. However, when training from scratch on small datasets, there is still a significant performance gap between ViTs and Convolutional Neural Networks (CNNs), which is attributed to the lack of inductive bias. To address this issue, we propose a Graph-based Vision Transformer (GvT) that utilizes graph convolutional projection and graph-pooling. In each block, queries and keys are calculated through graph convolutional projection based on the spatial adjacency matrix, while dot-product attention is used in another graph convolution to generate values. When using more attention heads, the queries and keys become lower-dimensional, making their dot product an uninformative matching function. To overcome this low-rank bottleneck in attention heads, we employ talking-heads technology based on bilinear pooled features and sparse selection of attention tensors. This allows interaction among filtered attention scores and enables each attention mechanism to depend on all queries and keys. Additionally, we apply graph-pooling between two intermediate blocks to reduce the number of tokens and aggregate semantic information more effectively. Our experimental results show that GvT produces comparable or superior outcomes to deep convolutional networks and surpasses vision transformers without pre-training on large datasets. 

# Graph-based Vision Transformer (GvT)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)
![PyTorch 1.9+](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)

Implementation of "Bridging the Gap with Convolutional Networks: A Graph-based Vision Transformer with Sparsity for Training on Small Datasets from Scratch" (Journal of Latex Class Files, 2025)

![GvT Architecture](https://via.placeholder.com/800x400.png?text=GvT+Architecture+Diagram+from+Paper)

## Key Features
✅ **Graph Convolutional Projection** for enhanced local feature learning  
✅ **Talking-heads Attention** with sparse tensor selection to overcome low-rank bottlenecks  
✅ **Adaptive Graph-pooling** for semantic information aggregation  
✅ **Small Dataset Optimization** - achieves SOTA on multiple benchmarks without pre-training  
✅ **Efficient Computation** - 21.4M FLOPs with comparable speed to CNNs

## Installation
```bash
conda create -n gvt python=3.8
conda activate gvt
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

