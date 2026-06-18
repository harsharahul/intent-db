# References

IntentDB's design builds on prior work in metric learning, instruction-conditioned
retrieval, embedding-space geometry, pseudo-relevance feedback, score fusion, and
query-log intent mining. The works below are the main influences, grouped by area.

## Metric learning and embedding adaptation
- Xing, Ng, Jordan & Russell. Distance metric learning. NIPS 2002. https://proceedings.neurips.cc/paper/2002/hash/c3e4035af2a1cde9f21e1ae1951ac80b-Abstract.html
- Schultz & Joachims. Learning a distance metric from relative comparisons. NIPS 2003. https://papers.nips.cc/paper/2366-learning-a-distance-metric-from-relative-comparisons
- Davis, Kulis, Jain, Sra & Dhillon. Information-theoretic metric learning (ITML). ICML 2007. https://www.cs.utexas.edu/~inderjit/public_papers/itml_icml07.pdf
- Jain, Kulis, Davis & Dhillon. Metric and kernel learning using a linear transformation. JMLR 2012. https://arxiv.org/abs/0910.5932
- Verma & Branson. Sample complexity of metric learning. NIPS 2015. https://arxiv.org/abs/1505.02729
- Gu, Li & Han. Generalized Fisher score for feature selection. UAI 2011. https://arxiv.org/pdf/1202.3725
- Veit, Belongie & Karaletsos. Conditional Similarity Networks. CVPR 2017. https://arxiv.org/abs/1603.07810
- Yoon et al. Search-Adaptor. ACL 2024. https://aclanthology.org/2024.acl-long.661/
- Vejendla. Drift-Adapter. EMNLP 2025. https://arxiv.org/abs/2509.23471
- Chroma. Embedding adapters. https://research.trychroma.com/embedding-adapters

## Embedding-space geometry
- Ethayarajh. How contextual are contextualized word representations? EMNLP 2019. https://aclanthology.org/D19-1006/
- Timkey & van Schijndel. Rogue dimensions. EMNLP 2021. https://aclanthology.org/2021.emnlp-main.372/
- Su et al. Whitening sentence representations. 2021. https://arxiv.org/abs/2103.15316
- Weller et al. On the theoretical limitations of embedding-based retrieval. ICLR 2026. https://arxiv.org/abs/2508.21038

## Instruction-conditioned retrieval
- Su et al. INSTRUCTOR. ACL 2023. https://arxiv.org/abs/2212.09741
- Asai et al. TART. https://arxiv.org/abs/2211.09260
- Weller et al. Promptriever. https://arxiv.org/abs/2409.11136
- Muennighoff et al. GritLM. ICLR 2024. https://arxiv.org/abs/2402.09906
- Wang et al. E5-Mistral. ACL 2024. https://arxiv.org/abs/2401.00368
- Weller et al. FollowIR. https://arxiv.org/abs/2403.15246
- Oh et al. InstructIR. https://arxiv.org/abs/2402.14334

## Pseudo-relevance feedback and fusion
- Rocchio. Relevance feedback in information retrieval. 1971.
- Li et al. Vector pseudo-relevance feedback. TOIS 2023. https://arxiv.org/abs/2108.11044
- Cormack, Clarke & Buettcher. Reciprocal rank fusion. SIGIR 2009. https://dl.acm.org/doi/10.1145/1571941.1572114
- Bruch, Gai & Ingber. An analysis of fusion functions for hybrid retrieval. TOIS 2023. https://arxiv.org/abs/2210.11934
- Gao et al. HyDE. ACL 2023. https://arxiv.org/abs/2212.10496

## Intent mining and evaluation
- Beeferman & Berger. Agglomerative clustering of a search engine query log. KDD 2000. https://dl.acm.org/doi/10.1145/347090.347176
- Wan et al. TnT-LLM. KDD 2024. https://arxiv.org/abs/2403.12173
- Thakur et al. BEIR. NeurIPS 2021. https://arxiv.org/abs/2104.08663
- Muennighoff et al. MTEB. EACL 2023. https://arxiv.org/abs/2210.07316
- Smucker, Allan & Carterette. Statistical significance tests for IR. CIKM 2007. https://dl.acm.org/doi/10.1145/1321440.1321528
- Urbano, Marrero & Martín. Statistical testing in IR. SIGIR 2019. https://arxiv.org/pdf/1905.11096
