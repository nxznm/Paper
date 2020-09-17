# Misc
1. [Query2box](https://openreview.net/forum?id=BJgr4kSFDS)'ICLR2020, which uses box embeddings to reasoning over KGs in vector space.
# Data augmentation
## Focus on graph
1. [NodeAug](https://dl.acm.org/doi/pdf/10.1145/3394486.3403063)'KDD2020, which does data augmentation(change attributes of related nodes and change graph structure by adding or removing edges) on each node separately, and uses subgraph mini-batch training(subgraph can be seen as a receptive field); this work focuses on the semi-supervised node classification task.
# KG embedding
## Traditional
1. [TransC](https://www.aclweb.org/anthology/D18-1222.pdf)'EMNLP2018, which distinguish concepts and instances in KGs differently. It uses a sphere to embed a concept. 
## KG reasoning
1. [MetaKGR](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/emnlp2019_meta.pdf)'EMNLP2019, which uses reinforcement learning to do KG reasoning(given a query (h,r,?), return a path as an explain to (h,r,t)), and combines meta-learning to alleviate few-short relations.
## Misc
1. [Fact Validation with KG embeddings](http://ceur-ws.org/Vol-2456/paper33.pdf), which uses KG embeddings as features, then trains by random forest with these features to do fact validation.
