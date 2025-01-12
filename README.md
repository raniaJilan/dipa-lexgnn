# LEX-GNN

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lex-gnn-label-exploring-graph-neural-network/fraud-detection-on-amazon-fraud)](https://paperswithcode.com/sota/fraud-detection-on-amazon-fraud?p=lex-gnn-label-exploring-graph-neural-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lex-gnn-label-exploring-graph-neural-network/fraud-detection-on-yelp-fraud)](https://paperswithcode.com/sota/fraud-detection-on-yelp-fraud?p=lex-gnn-label-exploring-graph-neural-network)

The author implementation of the [CIKM 2024](https://cikm2024.org/) short paper:  
**"LEX-GNN: Label-Exploring Graph Neural Network for Accurate Fraud Detection"**.  
[[Paper](https://dl.acm.org/doi/10.1145/3627673.3679956)] [[Poster](./lex_poster.pdf)]


[Woochang Hyun](https://scholar.google.com/citations?user=lswcPDIAAAAJ), 
Insoo Lee, 
[Bongwon Suh](https://scholar.google.com/citations?user=-nlhtEkAAAAJ)


## Overview

<p align="center">
    <a href="https://github.com/wdhyun/LEX">
        <img src="./lex_overview.png" width="750"/>
    </a>
<p>

**L**abel-**Ex**ploring **G**raph **N**eural **N**etwork (**LEX-GNN**) is a GNN-based fraud detector that predicts the fraud likelihood of nodes in a semi-supervised manner and adaptively adjusts the message passing pipeline for enhanced detection.


## Usage
- Requirements: `Torch` and `DGL`
- Dataset: `Yelp` and `Amazon` are loaded from [dgl.data.fraud](https://docs.dgl.ai/api/python/dgl.data.html#node-prediction-datasets) upon code execution.
- Run: `python main.py`

## Citation

```bibtex
@inproceedings{hyun2024lex,
  title={LEX-GNN: Label-Exploring Graph Neural Network for Accurate Fraud Detection},
  author={Hyun, Woochang and Lee, Insoo and Suh, Bongwon},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM'24)},
  year={2024}
}
```
