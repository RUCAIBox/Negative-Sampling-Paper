Negative-Sampling-Paper
================

This repository collects 100 papers related to negative sampling methods, covering multiple research fields such as 
Recommendation Systems (**RS**), Computer Vision (**CV**)，Natural Language Processing (**NLP**) and Contrastive Learning (**CL**).

Existing negative sampling methods can be roughly divided into five categories: **Static Negative Sampling**, **Hard Negative Sampling**, **Adversarial Sampling**, **Graph-based Sampling** and **Additional data enhanced Sampling**.

- [Category](#Category)
  - [Static Negative Sampling](#static-negative-sampling)
  - [Hard Negative Sampling](#hard-negative-sampling)
  - [Adversarial Sampling](#adversarial-sampling)
  - [Graph-based Sampling](#graph-based-sampling)
  - [Additional data enhanced Sampling](#additional-data-enhanced-sampling)

- [Future Outlook](#Future-Outlook)
  - [False Negative Problem](#false-negative-problem)
  - [Curriculum Learning](#curriculum-learning)
  - [Negative Sampling Ratio](#negative-sampling-ratio)
  - [Debiased Sampling](#debiased-sampling)
  - [Non-Sampling](#non-sampling)

Category
----
### Static Negative Sampling

-	BPR: Bayesian Personalized Ranking from Implicit Feedback. `UAI(2009)` **[RS]** **[[PDF](https://arxiv.org/pdf/1205.2618.pdf)]**

-	Real-Time Top-N Recommendation in Social Streams. `RecSys(2012)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2365952.2365968)]**

-	Distributed Representations of Words and Phrases and their Compositionality. `NIPS(2013)` **[NLP]** **[[PDF](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-com.pdf)]**

-	word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method. `arXiv(2014)` **[NLP]** **[[PDF](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-com.pdf)]**

-	Deepwalk: Online learning of social representations. `KDD(2014)` **[GRL]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2623330.2623732)]**

-	LINE: Large-scale Information Network Embedding. `WWW(2015)` **[GRL]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2736277.2741093)]**

-	Context- and Content-aware Embeddings for Query Rewriting in Sponsored Search. `SIGIR(2015)` **[NLP]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2766462.2767709)]**

-	node2vec: Scalable Feature Learning for Networks. `KDD(2016)` **[NLP]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2939672.2939754)]**

-	Fast Matrix Factorization for Online Recommendation with Implicit Feedback. `SIGIR(2016)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2911451.2911489)]**

-	Word2vec applied to Recommendation: Hyperparameters Matter. `RecSys(2018)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3240323.3240377)]**

-	General Knowledge Embedded Image Representation Learning. `TMM(2018)` **[CV]** **[[PDF](http://119.28.72.117/papers/TMM-KnowRep.pdf)]**

-	Alleviating Cold-Start Problems in Recommendation through Pseudo-Labelling over Knowledge Graph. `WSDM(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3437963.3441773)]**


### Hard Negative Sampling

-	Example-based learning for view-based human face detection. `TPAMI(1998)` **[CV]** **[[PDF](https://apps.dtic.mil/sti/pdfs/ADA295738.pdf)]**

-	Adaptive Importance Sampling to Accelerate Training of a Neural Probabilistic Language Model. `T-NN(2008)` **[NLP]** **[[PDF](https://infoscience.epfl.ch/record/82914/files/rr-03-35.pdf)]**

-	Optimizing Top-N Collaborative Filtering via Dynamic Negative Item Sampling. `SIGIR(2013)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2484028.2484126)]**

-	Bootstrapping Visual Categorization With Relevant Negatives. `TMM(2013)` **[CV]** **[[PDF](https://core.ac.uk/download/pdf/190707126.pdf)]**

-	Improving Pairwise Learning for Item Recommendation from Implicit Feedback. `WSDM(2014)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2556195.2556248)]**

-	Improving Latent Factor Models via Personalized Feature Projection for One Class Recommendation. `CIKM(2015)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2806416.2806511)]**

-	Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks. `CIKM(2016)` **[NLP]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2983323.2983872)]**

-	RankMBPR: Rank-aware Mutual Bayesian Personalized Ranking for Item Recommendation. `WAIM(2016)` **[RS]** **[[PDF](http://www.junminghuang.com/WAIM2016-yu.pdf)]**

-	Training Region-Based Object Detectors With Online Hard Example Mining. `CVPR(2016)` **[CV]** **[[PDF](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)]**

-	Hard Negative Mining for Metric Learning Based Zero-Shot Classification. `ECCV(2016)` **[ML]** **[[PDF](https://link.springer.com/content/pdf/10.1007/978-3-319-49409-8_45.pdf)]**

-	Vehicle detection in aerial images based on region convolutional neural networks and hard negative example mining. `Sensors(2017)` **[CV]** **[[PDF](https://www.mdpi.com/1424-8220/17/2/336/pdf)]**

-	WalkRanker: A Unified Pairwise Ranking Model with Multiple Relations for Item Recommendation. `AAAI(2018)` **[RS]** **[[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/11866/11725)]**

-	Bootstrapping Entity Alignment with Knowledge Graph Embedding. `IJCAI(2018)` **[KGE]** **[[PDF](https://www.ijcai.org/Proceedings/2018/0611.pdf)]**

-	Improving Occlusion and Hard Negative Handling for Single-Stage Pedestrian Detectors. `CVPR(2018)` **[CV]** **[[PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Noh_Improving_Occlusion_and_CVPR_2018_paper.pdf)]**

-	NSCaching: Simple and Efficient Negative Sampling for Knowledge Graph Embedding. `ICDE(2019)` **[KGE]** **[[PDF](https://arxiv.org/pdf/1812.06410)]**

-	Meta-Transfer Learning for Few-Shot Learning. `CVPR(2019)` **[CV]** **[[PDF](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Meta-Transfer_Learning_for_Few-Shot_Learning_CVPR_2019_paper.pdf)]**

-	ULDor: A Universal Lesion Detector for CT Scans with Pseudo Masks and Hard Negative Example Mining. `ISBI(2019)` **[CV]** **[[PDF](https://arxiv.org/pdf/1901.06359.pdf)]**

-	Distributed representation learning via node2vec for implicit feedback recommendation. `NCA(2020)` **[NLP]** **[[PDF](https://link.springer.com/article/10.1007/s00521-018-03964-2)]**

-	Simplify and Robustify Negative Sampling for Implicit Collaborative Filtering. `arXiv(2020)`  **[RS]** **[[PDF](https://arxiv.org/pdf/2009.03376)]**

-	Hard Negative Mixing for Contrastive Learning. `arXiv(2020)` **[CL]** **[[PDF](https://arxiv.org/pdf/2010.01028)]**

-	Bundle Recommendation with Graph Convolutional Networks. `SIGIR(2020)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3397271.3401198)]**

-	Supervised Contrastive Learning. `NIPS(2020)` **[CL]** **[[PDF](https://arxiv.org/abs/2004.11362)]**

-	Curriculum Meta-Learning for Next POI Recommendation. `KDD(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3447548.3467132)]**

-	Boosting the Speed of Entity Alignment 10×: Dual Attention Matching Network with Normalized Hard Sample Mining. `WWW(2021)` **[KGE]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3442381.3449897)]**

-	Hard-Negatives or Non-Negatives? A Hard-Negative Selection Strategy for Cross-Modal Retrieval Using the Improved Marginal Ranking Loss. `ICCV(2021)` **[CV]** **[[PDF](https://openaccess.thecvf.com/content/ICCV2021W/ViRaL/papers/Galanopoulos_Hard-Negatives_or_Non-Negatives_A_Hard-Negative_Selection_Strategy_for_Cross-Modal_Retrieval_ICCVW_2021_paper.pdf)]**


### Adversarial Sampling

-	Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks. `NIPS(2015)` **[CV]** **[[PDF](https://arxiv.org/pdf/1506.05751.pdf)]**

-	IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models. `SIGIR(2017)` **[IR]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3077136.3080786)]**

-	SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient. `AAAI(2017)` **[NLP]** **[[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/10804/10663)]**

-	KBGAN: Adversarial Learning for Knowledge Graph Embeddings. `NAACL(2018)` **[KGE]** **[[PDF](https://www.aclweb.org/anthology/N18-1133.pdf)]**

-	Neural Memory Streaming Recommender Networks with Adversarial Training. `KDD(2018)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3219819.3220004)]**

-	GraphGAN: Graph Representation Learning with Generative Adversarial Nets. `AAAI(2018)` **[GRL]** **[[PDF](https://ojs.aaai.org/index.php/AAAI/article/download/11872/11731)]**

-	CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks. `CIKM(2018)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3269206.3271743)]**

-	Adversarial Contrastive Estimation. `ACL(2018)` **[NLP]** **[[PDF](https://arxiv.org/pdf/1805.03642)]**

-	Incorporating GAN for Negative Sampling in Knowledge Representation Learning. `AAAI(2018)` **[KGE]** **[[PDF](https://ojs.aaai.org/index.php/AAAI/article/download/11536/11395)]**

-	Exploring the potential of conditional adversarial networks for optical and SAR image matching. `IEEE J-STARS(2018)` **[CV]** **[[PDF](https://elib.dlr.de/118413/1/FINAL%20VERSION_elib.pdf)]**

-	Deep Adversarial Metric Learning. `CVPR(2018)` **[CV]** **[[PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf)]**

-	Adversarial Detection with Model Interpretation. `KDD(2018)` **[ML]** **[[PDF](https://people.engr.tamu.edu/xiahu/papers/kdd18liu.pdf)]**

-	Adversarial Sampling and Training for Semi-Supervised Information Retrieval. `WWW(2019)` **[IR]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3308558.3313416)]**

-	Deep Adversarial Social Recommendation. `IJCAI(2019)` **[RS]** **[[PDF](https://www.ijcai.org/proceedings/2019/0187.pdf)]**

-	Adversarial Learning on Heterogeneous Information Networks. `KDD(2019)` **[HIN]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3292500.3330970)]**

-	Regularized Adversarial Sampling and Deep Time-aware Attention for Click-Through Rate Prediction. `CIKM(2019)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3357384.3357936)]**

-	Adversarial Knowledge Representation Learning Without External Model. `IEEE Access(2019)` **[KGE]** **[[PDF](https://ieeexplore.ieee.org/iel7/6287639/6514899/08599182.pdf)]**

-	Adversarial Binary Collaborative Filtering for Implicit Feedback. `AAAI(2019)` **[RS]** **[[PDF](https://ojs.aaai.org/index.php/AAAI/article/download/4460/4338)]**

-	ProGAN: Network Embedding via Proximity Generative Adversarial Network. `KDD(2019)` **[GRL]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3292500.3330866)]**

-	Generating Fluent Adversarial Examples for Natural Languages. `ACL(2019)` **[NLP]** **[[PDF](https://www.aclweb.org/anthology/P19-1559.pdf)]**

-	IPGAN: Generating Informative Item Pairs by Adversarial Sampling. `TNLLS(2020)`  **[RS]** **[[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9240960)]**

-	Contrastive Learning with Adversarial Examples. `arXiv(2020)` **[CL]** **[[PDF](https://arxiv.org/pdf/2010.12050)]**

-	PURE: Positive-Unlabeled Recommendation with Generative Adversarial Network. `KDD(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3447548.3467234)]**

-	Negative Sampling for Knowledge Graph Completion Based on Generative Adversarial Network. `ICCCI(2021)` **[KGE]** **[[PDF](https://link.springer.com/chapter/10.1007/978-3-030-88081-1_1)]**

-	Synthesizing Adversarial Negative Responses for Robust Response Ranking and Evaluation. `arXiv(2021)` **[NLP]** **[[PDF](https://arxiv.org/pdf/2106.05894)]**

-	Adversarial Feature Translation for Multi-domain Recommendation. `KDD(2021)` **[RS]** **[[PDF](http://nlp.csai.tsinghua.edu.cn/~xrb/publications/KDD-2021_AFT.pdf)]**

-	Adversarial training regularization for negative sampling based network embedding. `Information Sciences(2021)` **[GRL]** **[[PDF](https://doi.org/10.1016/j.ins.2021.07.018)]**

-	Adversarial Caching Training: Unsupervised Inductive Network Representation Learning on Large-Scale Graphs. `TNNLS(2021)` **[GRL]** **[[PDF](https://ieeexplore.ieee.org/abstract/document/9451538/)]**

-	A Robust and Generalized Framework for Adversarial Graph Embedding. `arxiv(2021)` **[GRL]** **[[PDF](https://arxiv.org/pdf/2105.10651)]**

-	Instance-wise Hard Negative Example Generation for Contrastive Learning in Unpaired Image-to-Image Translation. `ICCV(2021)` **[CV]** **[[PDF](http://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Instance-Wise_Hard_Negative_Example_Generation_for_Contrastive_Learning_in_Unpaired_ICCV_2021_paper.pdf)]**


### Graph-based Sampling

-	ACRec: a co-authorship based random walk model for academic collaboration recommendation. `WWW(2014)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2567948.2579034)]**

-	GNEG: Graph-Based Negative Sampling for word2vec. `ACL(2018)` **[NLP]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3219819.3219890)]**

-	Graph Convolutional Neural Networks for Web-Scale Recommender Systems. `KDD(2018)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3219819.3219890)]**

-	SamWalker: Social Recommendation with Informative Sampling Strategy. `WWW(2019)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3308558.3313582)]**

-	Understanding Negative Sampling in Graph Representation Learning. `KDD(2020)` **[GRL]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3394486.3403218)]**

-	Reinforced Negative Sampling over Knowledge Graph for Recommendation. `WWW(2020)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3366423.3380098)]**

-	MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems. `KDD(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3447548.3467408)]**

-	SamWalker++: recommendation with informative sampling strategy. `TKDE(2021)` **[RS]** **[[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9507306)]**

-	DSKReG: Differentiable Sampling on Knowledge Graph for Recommendation with Relational GNN. `CIKM(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3459637.3482092)]**


### Additional data enhanced Sampling

-	Leveraging Social Connections to Improve Personalized Ranking for Collaborative Filtering. `CIKM(2014)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2661829.2661998)]**

-	Social Recommendation with Strong and Weak Ties. `CIKM(2016)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2983323.2983701)]**

-	Bayesian Personalized Ranking with Multi-Channel User Feedback. `RecSys(2016)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2959100.2959163)]**

-	Joint Geo-Spatial Preference and Pairwise Ranking for Point-of-Interest Recommendation. `ICTAI(2017)` **[RS]** **[[PDF](https://www.researchgate.net/profile/Fajie-Yuan/publication/308501951_Joint_Geo-Spatial_Preference_and_Pairwise_Ranking_for_Point-of-Interest_Recommendation/links/59bc0406aca272aff2d47bda/Joint-Geo-Spatial-Preference-and-Pairwise-Ranking-for-Point-of-Interest-Recommendation.pdf)]**

-	A Personalised Ranking Framework with Multiple Sampling Criteria for Venue Recommendation. `CIKM(2017)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3132847.3132985)]**

-	An Improved Sampling for Bayesian Personalized Ranking by Leveraging View Data. `WWW(2018)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3184558.3186905)]**

-	Reinforced Negative Sampling for Recommendation with Exposure Data. `IJCAI(2019)` **[RS]** **[[PDF](https://www.ijcai.org/Proceedings/2019/0309.pdf)]**

-	Geo-ALM: POI Recommendation by Fusing Geographical Information and Adversarial Learning Mechanism. `IJCAI(2019)` **[RS]** **[[PDF](https://www.ijcai.org/Proceedings/2019/0250.pdf)]**

-	Bayesian Deep Learning with Trust and Distrust in Recommendation Systems. `WI(2019)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3350546.3352496)]**

-	Socially-Aware Self-Supervised Tri-Training for Recommendation. `arXiv(2021)` **[RS]** **[[PDF](https://arxiv.org/pdf/2106.03569)]**

-	DGCN: Diversified Recommendation with Graph Convolutional Networks. `WWW(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3442381.3449835)]**


Future Outlook
----
### False Negative Problem

-	Incremental False Negative Detection for Contrastive Learning. `arXiv(2021)` **[CL]** **[[PDF](https://arxiv.org/pdf/2106.03719)]**

-	Graph Debiased Contrastive Learning with Joint Representation Clustering. `IJCAI(2021)` **[GRL & CL]** **[[PDF](https://www.ijcai.org/proceedings/2021/0473.pdf)]**

-	Relation-aware Graph Attention Model With Adaptive Self-adversarial Training. `AAAI(2021)` **[KGE]** **[[PDF](https://www.aaai.org/AAAI21Papers/AAAI-5774.QinX.pdf)]**


### Curriculum Learning

-	On The Power of Curriculum Learning in Training Deep Networks. `ICML(2016)` **[CV]** **[[PDF](http://proceedings.mlr.press/v97/hacohen19a/hacohen19a.pdf)]**

-	Graph Representation with Curriculum Contrastive Learning. `IJCAI(2021)` **[GRL & CL]** **[[PDF](https://www.ijcai.org/proceedings/2021/0317.pdf)]**

### Negative Sampling Ratio

-	Are all negatives created equal in contrastive instance discrimination. `arXiv(2020)` **[CL]** **[[PDF](https://arxiv.org/pdf/2010.06682)]**

-	SimpleX: A Simple and Strong Baseline for Collaborative Filtering. `CIKM(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3459637.3482297)]**

-	Rethinking InfoNCE: How Many Negative Samples Do You Need. `arXiv(2021)` **[CL]** **[[PDF](https://arxiv.org/pdf/2105.13003.pdf)]**

### Debiased Sampling

-	Debiased Contrastive Learning. `NIPS(2020)` **[CL]** **[[PDF](https://arxiv.org/pdf/2007.00224)]**

-	Contrastive Learning for Debiased Candidate Generation in Large-Scale Recommender Systems. `KDD(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3447548.3467102)]**

### Non-Sampling

-	Beyond Hard Negative Mining: Efficient Detector Learning via Block-Circulant Decomposition. `ICCV(2013)` **[CV]** **[[PDF](http://openaccess.thecvf.com/content_iccv_2013/papers/Henriques_Beyond_Hard_Negative_2013_ICCV_paper.pdf)]**

-	Efficient Heterogeneous Collaborative Filtering without Negative Sampling for Recommendation. `AAAI(2020)` **[RS]** **[[PDF](https://ojs.aaai.org/index.php/AAAI/article/download/5329/5185)]**

-	Efficient Non-Sampling Knowledge Graph Embedding. `WWW(2021)` **[KGE]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3442381.3449859)]**
