# Classic papers
1. NCE'AISTATS2010
2. word2vec
# Misc
1. [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181.pdf)'EMNLP2014
2. [PROBABILITY CALIBRATION FOR KNOWLEDGE GRAPH EMBEDDING MODELS](https://openreview.net/pdf?id=S1g8K1BFwS)'ICLR2020
3. [Counterfactual Vision-and-Language Navigation via Adversarial Path Sampler](https://arxiv.org/pdf/1911.07308.pdf)'ECCV2020
4. [Logic Constrained Pointer Networks for Interpretable Textual Similarity](https://www.ijcai.org/Proceedings/2020/0333.pdf)'IJCAI2020
# Recommendation
1. KGAT'KDD2019
# Meta learning
## CV
1. [Meta-Weight-Net](https://arxiv.org/pdf/1902.07379.pdf)'NIPS2019
## KG
1. [MetaKGR](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/emnlp2019_meta.pdf)'EMNLP2019
# Data augmentation
## Focus on graph
1. [NodeAug](https://dl.acm.org/doi/pdf/10.1145/3394486.3403063)'KDD2020, which does data augmentation(change attributes of related nodes and change graph structure by adding or removing edges) on each node separately, and uses subgraph mini-batch training(subgraph can be seen as a receptive field); this work focuses on the semi-supervised node classification task.
## Focus on text
1. [Learning beyond datasets: Knowledge Graph Augmented Neural Networks for Natural language Processing](https://www.aclweb.org/anthology/N18-1029.pdf)'NAACL2018, which uses KG to do data augmentation for NLP.
2. [Iterative Paraphrastic Augmentation with Discriminative Span Alignment](https://arxiv.org/pdf/2007.00320.pdf)'Arxiv2020
3. [Can We Achieve More with Less? Exploring Data Augmentation for Toxic Comment Classification](https://arxiv.org/pdf/2007.00875.pdf)'Arxiv2020
# KG embedding
## Traditional
1. TransE'NIPS2013
2. TransR'AAAI2015
3. QuatE'NIPS2019
## Use rules
1. [TransE-RW](https://geog.ucsb.edu/~jano/2018-EKAW18_TransRW.pdf)'EKAW2018
2. IterE'WWW2019
3. RPJE'AAAI2020
4. [UniKER](https://grlplus.github.io/papers/84.pdf)'ICML-Workshop2020
## Learn rules(there is intersection with KG reasoning)
1. [RuLES](https://people.mpi-inf.mpg.de/~dstepano/conferences/ISWC2018/paper/ISWC2018paper.pdf)'ISWC2018
## Use attributes
1. [KR-EAR](https://www.ijcai.org/Proceedings/16/Papers/407.pdf)'IJCAI2016
2. [TransEA](https://www.aclweb.org/anthology/W18-3017.pdf)'ACL-Workshop2018
## KG reasoning
1. PRA'EMNLP2011
2. Neural-LP'NIPS2017
3. [Multi-Hop](https://arxiv.org/pdf/1808.10568.pdf)'EMNLP2018
4. [MetaKGR](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/emnlp2019_meta.pdf)'EMNLP2019, which uses reinforcement learning to do KG reasoning(given a query (h,r,?), return a path as an explain to (h,r,t)), and combines meta-learning to alleviate few-short relations.
5. [Query2box](https://openreview.net/forum?id=BJgr4kSFDS)'ICLR2020, which uses box embeddings to reasoning over KGs in vector space.
6. [CBR](https://openreview.net/pdf?id=AEY9tRqlU7)'AKBC2020
## Entity alignment
1. BootEA'IJCAI2018
2. RSNs'ICML2019
## Cluster
1. [ExCut: Explainable Embedding-based Clustering over Knowledge Graphs](http://people.mpi-inf.mpg.de/~gadelrab/downloads/ExCut/excut_preprint.pdf)'ISWC2020, which iteratively does clusters by embeddings and learns rules as explanations for the clusters.
## Alert
1. [Knowledge Base Completion: Baselines Strike Back](https://www.aclweb.org/anthology/W17-2609.pdf)'ACL2017
2. [A Re-evaluation of Knowledge Graph Completion Methods](https://arxiv.org/pdf/1911.03903.pdf)'ACL2020
## Misc
1. [Sparsity and Noise: Where Knowledge Graph Embeddings Fall Short](https://www.aclweb.org/anthology/D17-1184.pdf)'ACL2017, which talks about the influence of sparsity and noise for KGE.
2. KBGAN'NAACL2018, which uses GAN to do negative sampling.
3. [Open-World Knowledge Graph Completion](https://arxiv.org/pdf/1711.03438.pdf)'AAAI2018, which does KG completion in open world(new entities and relations emerge).
4. [CKRL](https://arxiv.org/abs/1705.03202)'AAAI2018, which assumes that triples in KGs are not always right(may have some noise), and triples should be treated differently(each triple has a distinct confidence). I think this work is similar to [TransE-RW](https://geog.ucsb.edu/~jano/2018-EKAW18_TransRW.pdf)'EKAW2018, the diffence is that TransE-RW uses rules to calculate confidence, while CKRL models the confidence more complicated.
5. [TransC](https://www.aclweb.org/anthology/D18-1222.pdf)'EMNLP2018, which distinguish concepts and instances in KGs differently. It uses a sphere to embed a concept. 
6. [Fact Validation with KG embeddings](http://ceur-ws.org/Vol-2456/paper33.pdf)'2020, which uses KG embeddings as features, then trains by random forest with these features to do fact validation.
