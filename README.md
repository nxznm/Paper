# Classic papers
* NCE'AISTATS2010
* word2vec
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762v5.pdf)'NIPS2017
* [Bert](https://arxiv.org/pdf/1810.04805.pdf), which deserves reading more times!
* [How to represent part-whole hierarchies in a neural network](https://arxiv.org/pdf/2102.12627.pdf)'Arxiv2021, which proposed by *Hinton*. I think it is a appropriately new trial in Neural Symbolic Learning (use neural networks to convert a image into a parse tree). The whole techiniques are composed with transformers, neural fields, contrastive representation learning, distillation and capsules.
# Not read but the idea is interesting
* [Import2vec Learning Embeddings for Software Libraries](https://arxiv.org/pdf/1904.03990.pdf)'Arxiv2019
# Misc
* [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181.pdf)'EMNLP2014
* [PROBABILITY CALIBRATION FOR KNOWLEDGE GRAPH EMBEDDING MODELS](https://openreview.net/pdf?id=S1g8K1BFwS)'ICLR2020
* [Counterfactual Vision-and-Language Navigation via Adversarial Path Sampler](https://arxiv.org/pdf/1911.07308.pdf)'ECCV2020
* [Logic Constrained Pointer Networks for Interpretable Textual Similarity](https://www.ijcai.org/Proceedings/2020/0333.pdf)'IJCAI2020
* [Graph Structure of Neural Networks](https://arxiv.org/pdf/2007.06559.pdf)'ICML2020, proposed by Jiaxuan You. It represents neural networks as graphs of connections between neurons, and depicts the way how does the graph structure of neural networks affect their predictive performance.
# Mechine learning
* [Regularizing Recurrent Neural Networks via Sequence Mixup](https://arxiv.org/pdf/2012.07527.pdf), it adopts several regularization techniques from feed-forward networks into RNN. I don't totally understand the paper (maybe re-read in future).
# Deep learning
## Transformer
* [Self-Attention with Relative Position Representations](https://www.aclweb.org/anthology/N18-2074.pdf)'NAACL-HLT2018, which changes the absolute position embedding in original transformer to relative position embedding.
* [Enhancing the Transformer With Explicit Relational Encoding for Math Problem Solving](https://arxiv.org/pdf/1910.06611.pdf)'Arxiv2020, it proposes a change in the attention mechanism of Transformer, i think it deserves re-reading.
* [RealFormer: Transformer Likes Residual Attention](https://arxiv.org/pdf/2012.11747.pdf)'Arxiv2020, which is preposed by Google, and it focuses on adding a residual connection on attention score (so simple!!!). And it says that Post-LN usually performs better than Pre-LN, but Post-LN needs warm up strategy, while Pre-LN does not need (we can set a large learning rate in Pre-LN), such opinion is proposed by [this paper](https://openreview.net/forum?id=B1x8anVFPr).
* [RETHINKING POSITIONAL ENCODING IN LANGUAGE PRE-TRAINING](https://arxiv.org/pdf/2006.15595.pdf)'Arxiv2020, this work is well-written!! It thinks that the direct way of adding position embedding to input embedding is not suitable, as the two information is heterogeneous. It also proposes a novel way to tackle `[CLS]` (I do not read this part very carefully). Anyway, it deserves re-reading!
# Recommendation
* [Knowledge Graph Convolutional Networks for Recommender Systems](https://arxiv.org/pdf/1904.12575.pdf)'WWW2019
* KGAT'KDD2019
# Meta learning
## CV
* [Meta-Weight-Net](https://arxiv.org/pdf/1902.07379.pdf)'NIPS2019
## KG
* [MetaKGR](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/emnlp2019_meta.pdf)'EMNLP2019
# Abduction Learning
* [Abduction and Argumentation for Explainable Machine Learning: A Position Survey](https://arxiv.org/pdf/2010.12896.pdf)'Arxiv2020
# Data augmentation
## Focus on graph
* [NodeAug](https://dl.acm.org/doi/pdf/10.1145/3394486.3403063)'KDD2020, which does data augmentation(change attributes of related nodes and change graph structure by adding or removing edges) on each node separately, and uses subgraph mini-batch training(subgraph can be seen as a receptive field); this work focuses on the semi-supervised node classification task.
## Focus on text
* [Learning beyond datasets: Knowledge Graph Augmented Neural Networks for Natural language Processing](https://www.aclweb.org/anthology/N18-1029.pdf)'NAACL2018, which uses KG to do data augmentation for NLP.
* [Iterative Paraphrastic Augmentation with Discriminative Span Alignment](https://arxiv.org/pdf/2007.00320.pdf)'Arxiv2020
* [Can We Achieve More with Less? Exploring Data Augmentation for Toxic Comment Classification](https://arxiv.org/pdf/2007.00875.pdf)'Arxiv2020
# Network embedding
* [DeepWalk](https://arxiv.org/pdf/1403.6652.pdf)'KDD2014, random walk + language model (why: the frequency distribution of vertices in random walks of social network and words in a language both follow a power law).
* [LINE](https://arxiv.org/pdf/1503.03578.pdf)'WWW2015, the designed objective function which preserves both the first-order and second-order proximities. It proposes an edge-sampling algorithm for optimizing the objective to improve the effectiveness and efficiency. 
* [HAN](https://arxiv.org/pdf/1903.07293.pdf)'WWW2019, which focuses on heterogeneous graphs. And it uses a hierarchical attention, including node-level and semantic-level attentions.
# KG embedding
## Traditional
* TransE'NIPS2013
* TransR'AAAI2015
* [ANALOG](https://arxiv.org/pdf/1705.02426.pdf)'ICML2017, which can degenerate into DisMult, ComplEx and HolE. It focuses on analogical structures in KGs, such as "man is king as woman is to queen", which man and woman, king and queen are analogies. ANALOG designs special constraints on relation matrixs, such that analogical structures can be held in the model.
* [R-GCN](https://arxiv.org/pdf/1703.06103.pdf)'ESWC2018
* [ConvKB](https://www.aclweb.org/anthology/N18-2053.pdf)'NAACL-HLT2018, it takes transitional characteristics into accounts by using CNN(similart to ConvE, both them use CNN, ConvE doesn't hold transitional characteristics, while ConvKB holds). Different from ConvE which uses CNN to obtain features from head and relation, ConvKB obtains features from head relation and tail simultaneously. Although ConvKB gets competitive results in KGC, some doubts have rasied to question the improvement, e.g. [A Re-evaluation of Knowledge Graph Completion Methods](https://arxiv.org/pdf/1911.03903.pdf)'ACL2020.
* QuatE'NIPS2019
* [SACN](https://arxiv.org/pdf/1811.04441.pdf)'AAAI2019, SACN = WGCN(weighted GCN) + Conv-TransE. It takes advantage of knowledge graph node connectivity(GCN), node attributes(add attribute nodes) and relation types(WGCN). Conv-TransE keeps the translational property between entities and relations to learn node embeddings for the link prediction(similar to ConvE, both them use 2D convolution, but ConvE doesn't hold the translational property, while Conv-TransE does).
* [VR-GCN](https://www.ijcai.org/Proceedings/2019/0574.pdf)'IJCAI2019, which generates both entity embeddings and relation embeddings simultaneously. VR-GCN is capable of learning the vectorized embedding of relations, in comparison with existing GCNs. 
* [QUATRE](https://arxiv.org/pdf/2009.12517.pdf)'Arxiv2020, i think it combines QuatE and TransR together.
* [HAKE](https://arxiv.org/pdf/1911.09419.pdf)'AAAI2020, which combines the modulus (encode different categories) and phase (encode unique information in the same category) information. This method only uses triples, while it can caputure semantic hierarchy. I think this work is solid.
## Use rules
* [TransE-RW](https://geog.ucsb.edu/~jano/2018-EKAW18_TransRW.pdf)'EKAW2018
* IterE'WWW2019
* [Quantum Embedding of Knowledge for Reasoning](https://proceedings.neurips.cc/paper/2019/file/cb12d7f933e7d102c52231bf62b8a678-Paper.pdf)'NeurIPS2019, E2R, which encodes logical structrues (T-box and A-box) into a vector space wich quantum logic. I think idea behind the model is similar to many works (like Query2box) which encode some concept information (T-box) into embedding. However, I think this model (E2R) is more general, it can encode many logic information into embedding (compared with existing work). I like this work.
* RPJE'AAAI2020
* [UniKER](https://grlplus.github.io/papers/84.pdf)'ICML-Workshop2020
* [A Hybrid Model for Learning Embeddings and Logical Rules Simultaneously from Knowledge Graphs](https://arxiv.org/pdf/2009.10800.pdf)'Arxiv2020, it is similar to my first work. It iteratively learns rules and embeddings. At each iteration, learned embeddings help to prune the rules search space (special filter function using embeddings); and rules help to infer new facts (use importance sampling to sample from inferred facts, and then add into the training set). It is interesting that the experiment result is really good (compared to SOTA). So, why my method fails??? :(
## Use pathes
* [PTransE](https://www.aclweb.org/anthology/D15-1082.pdf)'EMNLP2015. If a path is h -r1-> e1 -r2-> t, then the objective is h + r1 = e1, e1 + r2 = t and h + (r1*r2) = t. As there are many paths and some paths are noise, so the paper proposes a metric to filter and uses the metric to do aggregation of the paths.
* RSNs'ICML2019
* [PPKE: Knowledge Representation Learning by Path-based Pre-training](https://arxiv.org/pdf/2012.03573.pdf)'Arxiv2020, which follows CoKE and does path-based pre-training (sample lots of paths and use transformers to capture the context of paths), then it will do fine-tune with specific downstream task (e.g. link prediction or relation prediction).
## Learn rules(there is intersection with KG reasoning)
* [RuLES](https://people.mpi-inf.mpg.de/~dstepano/conferences/ISWC2018/paper/ISWC2018paper.pdf)'ISWC2018
* [RLvLR](https://www.ijcai.org/Proceedings/2018/0297.pdf)'IJCAI2018, which is comparable with Neural-LP. It uses KG embeddings to accelerate the rule finding (also uses sampling to make the embedding model scalable to large KGs), and uses matrix multiplication to accelerate the rule filtering (more efficient to calculate standard confidence).
## Use attributes
* [KR-EAR](https://www.ijcai.org/Proceedings/16/Papers/407.pdf)'IJCAI2016
* [TransEA](https://www.aclweb.org/anthology/W18-3017.pdf)'ACL-Workshop2018
## Focus on context
* [KG-BERT](https://arxiv.org/pdf/1909.03193.pdf)'AAAI2020, which treats triples in knowledge graphs as textual sequences and uses bert to model these triples.
* [Entity Context and Relational Paths for Knowledge Graph Completion](https://arxiv.org/pdf/2002.06757.pdf)'Arxiv2020
* [CoKE: Contextualized Knowledge Graph Embedding](https://arxiv.org/pdf/1911.02168.pdf)'Arxiv2020, it employs transformer encoder to obtain contextualized representations (two types of graph contexts are studied: edges and paths).
* [HittER](https://arxiv.org/pdf/2008.12813.pdf)'Arxiv2020, which uses transformer to capture both the entity-relation and entity-context interactions. The interesting thing is that we can make an analogy between HittER with CompGCN (More generally, we can make an analogy between transformer and GCN).
* [Multi-Task Learning for Knowledge Graph Completion with Pre-trained Language Models](https://www.aclweb.org/anthology/2020.coling-main.153.pdf)'COLING2020, which follows KG-BERT, and uses multi-task learning (three tasks: link prediction, relation prediction and relevance ranking task) to combine linguistic information of pre-trained models and triple structural information.
* [RETRA: Recurrent Transformers for Learning Temporally Contextualized Knowledge Graph Embeddings](https://openreview.net/pdf?id=l7fvWxQ3RG)'ESWC2021-UnderReview, which focuses on temporally contextualized KGE by combining transformer and RNN. 
## KG reasoning
* PRA'EMNLP2011
* [Traversing Knowledge Graphs in Vector Space](https://arxiv.org/pdf/1506.01094.pdf)'EMNLP2015, it traverses in vector space to answer queries, and the paper shows that compositional training (modeling on path queries with length more than three) can improve knowledge base completion (sounds amazing!).
* Neural-LP'NIPS2017
* [Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks](https://www.aclweb.org/anthology/E17-1013.pdf)'EACL2017. Given a query, answer the relation between two entities. The whole task is similar to PRA, the difference is that this paper uses RNN and each path doesn't only have relations but also entities. 
* [Multi-Hop](https://arxiv.org/pdf/1808.10568.pdf)'EMNLP2018
* [Embedding Logical Queries on Knowledge Graphs](https://arxiv.org/pdf/1806.01445.pdf)'NIPS2018, which is followed by Query2box.
* [MetaKGR](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/emnlp2019_meta.pdf)'EMNLP2019, which uses reinforcement learning to do KG reasoning(given a query (h,r,?), return a path as an explain to (h,r,t)), and combines meta-learning to alleviate few-short relations.
* [Embed2Reason](http://papers.nips.cc/paper/8797-quantum-embedding-of-knowledge-for-reasoning.pdf)'NIPS2019, which embeds a symbolic KB into a vector space in a logical structure preserving manner (inspired by the theory of Quantum Logic).
* [Query2box](https://openreview.net/forum?id=BJgr4kSFDS)'ICLR2020, which uses box embeddings to reasoning over KGs in vector space.
* [CBR](https://openreview.net/pdf?id=AEY9tRqlU7)'AKBC2020
* [BetaE](https://arxiv.org/pdf/2010.11465.pdf)'NIPS2020, which is the first embedding-based method that could handle arbitrary FOL queries on KGs (Beta distributions + probabilistic logical operators). The paper is another work by the author of Query2box.
* [EM-RBR](https://openreview.net/pdf?id=EKw6nZ4QkJl)'Under_Review_ICLR2021, it utilizes relational background knowledge contained in rules to conduct multi-relation reasoning link prediction rather than superficial vector triangle linkage in embedding models. It solves completion through real rule-based reasoning (rather than uses rules to obtain better embeddings), sounds exciting!
## Entity alignment
* BootEA'IJCAI2018
* RSNs'ICML2019
* [Visual Pivoting for (Unsupervised) Entity Alignment](https://arxiv.org/pdf/2009.13603.pdf)'Arxiv2020(accepted by AAAI'2020), it focuses on multi-modal embedding learning, and considers auxiliary information including images, relations and attributes(mostly focuses on images). 
## OWL Ontologies Embedding (concepts/entity types & instances)
* [Learning Entity Type Embeddings for Knowledge Graph Completion](https://persagen.com/files/misc/Moon2017Learning.pdf)'CIKM2017, proposes a new task to predict the missing entity types.
* [Embedding OWL Ontologies with OWL2Vec*](http://ceur-ws.org/Vol-2456/paper9.pdf)'2019
* [OWL2Vec∗: Embedding of OWL Ontologies](https://arxiv.org/pdf/2009.14654.pdf)'2020, there are two paradigms of embedding, one is semantic embedding (e.g. TransE), the other is to first explicitly explore the neighborhoods of entities and relations in the graph, and then learn the embeddings using a language model (e.g. node2vec, rdf2vec). This paper  belongs to the language model paradigm, but preserves the semantics not only of the graph structure, but also of the lexical information and the logical constructors. Note that the graph of an ontology, which includes hierarchical categorization structure, differs from the multi-relation graph composed of role assertions of a typical KG; furthermore the ontology’s lexical information and logical constructors can not be successfully exploited by the aforementioned KG embedding methods.
## Cluster
* [ExCut: Explainable Embedding-based Clustering over Knowledge Graphs](http://people.mpi-inf.mpg.de/~gadelrab/downloads/ExCut/excut_preprint.pdf)'ISWC2020, which iteratively does clusters by embeddings and learns rules as explanations for the clusters.
## Alert
* [Knowledge Base Completion: Baselines Strike Back](https://www.aclweb.org/anthology/W17-2609.pdf)'ACL2017
* [A Re-evaluation of Knowledge Graph Completion Methods](https://arxiv.org/pdf/1911.03903.pdf)'ACL2020
* [On the Ambiguity of Rank-Based Evaluation of Entity Alignment or Link Prediction Methods](https://arxiv.org/pdf/2002.06914.pdf)'Arxiv2020, it proposes a new rank function which involves the situation that many scores are the same, and a new metric adjusted mean rank (compared with mean rank).
## Inductive setting
* [Inductively Representing Out-of-Knowledge-Graph Entities by Optimal Estimation Under Translational Assumptions](https://arxiv.org/pdf/2009.12765.pdf)'Arxiv2020, simple and straightforward.
##  Dynamic & Temporal setting
* [RECURRENT EVENT NETWORK: GLOBAL STRUCTURE INFERENCE OVER TEMPORAL KNOWLEDGE GRAPH](https://arxiv.org/pdf/1904.05530.pdf)'EMNLP2020
## Few-shot or Zero-shot setting
* [GMatching](https://www.aclweb.org/anthology/D18-1223.pdf)'EMNLP2018, which is the first research on few-shot learning for knowledge graphs. Techiniques: neightbor encoder (encoder an entity h with its neighbors (r,t)), matching processor (a LSTM module for multi-step matching, given two entity pair, output a similarity score)
* [MetaR](https://www.aclweb.org/anthology/D19-1431.pdf)'EMNLP2019, which follows the setting in GMatching. I think the idea belongs to the range of meta learning. It extracts relation meta information from few short train instances (easy), and calculates gradient meta information based on support set. It updates relation meta on querry set based on gradient meta from support set, and the final training objective is based on the loss on querry set (this part sounds confusing, but it is really similar to MAML (an algorithm of meta learning), both a bit confusing, Oh, this part seems really reasonable, I got it!).  
* [Adaptive Attentional Network for Few-Shot Knowledge Graph Completion](https://arxiv.org/pdf/2010.09638.pdf)'EMNLP2020. It learns dynamic/adaptive entity embeddings (entities exhibit diverse roles within task relations), and dynamic/adaptive reference embeddings (references make different contributions to queries). And it uses transformer encoder for entity pairs (reference / query).
## Misc
* [Sparsity and Noise: Where Knowledge Graph Embeddings Fall Short](https://www.aclweb.org/anthology/D17-1184.pdf)'ACL2017, which talks about the influence of sparsity and noise for KGE.
* KBGAN'NAACL2018, which uses GAN to do negative sampling.
* [Open-World Knowledge Graph Completion](https://arxiv.org/pdf/1711.03438.pdf)'AAAI2018, which does KG completion in open world(new entities and relations emerge).
* [CKRL](https://arxiv.org/abs/1705.03202)'AAAI2018, which assumes that triples in KGs are not always right(may have some noise), and triples should be treated differently(each triple has a distinct confidence). I think this work is similar to [TransE-RW](https://geog.ucsb.edu/~jano/2018-EKAW18_TransRW.pdf)'EKAW2018, the diffence is that TransE-RW uses rules to calculate confidence, while CKRL models the confidence more complicated.
* [TransC](https://www.aclweb.org/anthology/D18-1222.pdf)'EMNLP2018, which distinguish concepts and instances in KGs differently. It uses a sphere to embed a concept. 
* [Fact Validation with KG embeddings](http://ceur-ws.org/Vol-2456/paper33.pdf)'2020, which uses KG embeddings as features, then trains by random forest with these features to do fact validation.
## KGE Libraries
* [OpenKE](https://www.aclweb.org/anthology/D18-2024.pdf)'EMNLP2018, which separates a large-scale KG into several parts and adapt KE models for parallel training (thus capable of embedding large-scale KGs). And it proposes a novel negative sampling strategy (offset-based negative sampling algorithm, i don't understand the algorithm) for further acceleration.
* AmpliGraph'2019, it has no paper.
* [Pykg2vec](https://arxiv.org/pdf/1906.04239.pdf)'Arxiv2019
* [LibKGE](https://openreview.net/pdf?id=BkxSmlBFvr)'ICLR2020, it indicates that training strategies (loss function, negative sampling, e.t.c) have a significant impact on model performance and may account for a substantial fraction of the progress (rather than the model itself) made in recent years. Interesting and inspiring!
* [TorchKGE](https://arxiv.org/pdf/2009.02963.pdf)'IWKG-KDD2020, it evaluates much faster than OpenKGE.
* [PyKEEN](https://arxiv.org/pdf/2007.14175.pdf)'Arxiv2020
* [GraphVite](https://arxiv.org/pdf/1903.00757.pdf)'WWW2019, which accelarates node embedding greatly(can process very large scale) by designing a CPU-GPU hybrid system, focused on only one machine with mutiple CPU cores and multiple GPUs (one machine, multi-GPUs). 
* [PBG](https://arxiv.org/pdf/1903.12287.pdf)'SysML2019, distributed training (multi-machines, multi-GPUs).
* [DGL-KE](https://arxiv.org/pdf/2004.08532.pdf)'SIGIR2020, distributed training (multi-machines, multi-GPUs).
## Survey
* [Knowledge Graph Embedding: A Survey of Approaches and Applications](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8047276)'TKDE2017, it concludes very comprehensively and deserves re-reading (I just read until section 3.5). 
