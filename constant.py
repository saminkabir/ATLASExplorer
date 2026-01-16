description_intro = f"""

Similarity search enables efficient retrieval of vectors similar to a query and underpins applications such as retrieval-augmented generation, recommendation systems, and pattern recognition. Although approximate nearest neighbor search (ANNS) methods now dominate due to their strong efficiency–accuracy trade-offs, existing studies remain fragmented, often missing important algorithmic families, recent advances, diverse datasets, and rigorous statistical validation. ATLASExplorer is a web-based interactive engine that provides unified access to the most comprehensive ANNS benchmark to date, supporting statistical analysis of 43 methods across 58 heterogeneous datasets and intuitive visual exploration of accuracy–runtime trade-offs. The system further enables quantitative comparison from recall–runtime curves using the Weighted Cumulative Speedup over Recall (WCSR) measure, ensuring fair evaluation over overlapping operating regions. By integrating large-scale benchmarking, visual analytics, and statistically grounded evaluation in a single platform, ATLASExplorer offers a practical, reproducible, and user-friendly framework for analyzing, comparing, and selecting similarity search methods.


* Github repo: https://github.com/saminkabir/ATLAS-benchmark

"""

User_Manual = f"""

(a) Visualizing recall–throughput curves to analyze accuracy–efficiency trade-offs

(b) Identifying the best-performing algorithm at a target recall constraint

(c) Comparing algorithms at the dataset level using WCSR over the overlapped recall range

(d) Ranking algorithms across datasets using statistically grounded significance tests


"""


Contributors = f"""

#### Contributors

* [Ahmed Samin Yeaser Kabir](https://saminkabir.github.io/) (The Ohio State University)
* [John Paparrizos](https://www.paparrizos.org) (The Ohio State University)

"""








datasets=['glove', 'deep', 'imageNet', 'sift', 'MNIST', 'audio', 'notre', 'nuswide', 'ukbench', 'sun', 'millionSong']
datasets=['glove', 'deep', 'imageNet', 'sift', 'audio', 'notre', 'nuswide', 'sun', 'millionSong']
datasets=['ISC_EHB_DepthPhases', 'Iquique', 'MNIST', 'Meier2019JGR', 'Music', 'NEIC', 'OBS', 'OBST2024', 'PNW', 'Yelp', 'agnews-mxbai-1024-euclidean', 'arxiv-nomic-768-normalized', 'astro', 'audio', 'bigann', 'ccnews-nomic-768-normalized', 'celeba-resnet-2048-cosine', 'cifar', 'coco-nomic-768-normalized', 'codesearchnet-jina-768-cosine', 'crawl', 'deep', 'enron', 'ethz', 'geofon', 'gist', 'glove', 'gooaq-distilroberta-768-normalized', 'imageNet', 'imagenet-clip-512-normalized', 'instancegm', 'laion-clip-512-normalized', 'landmark-dino-768-cosine', 'landmark-nomic-768-normalized', 'lastfm', 'lendb', 'llama-128-ip', 'millionSong', 'movielens', 'netflix', 'notre', 'nuswide', 'nytimes', 'random', 'sald', 'seismic', 'sift', 'space', 'stead', 'sun', 'text', 'text-to-image', 'tiny5m', 'trevi', 'txed', 'ukbench', 'uqv', 'vcseis', 'word2vec', 'yahoo-minilm-384-normalized', 'yahoomusic', 'yandex-200-cosine']
datasets=['deep', 'imageNet', 'audio', 'glove', 'millionSong', 'notre', 'nuswide', 'sun', 'MNIST', 'sift', 'ukbench', 'Yelp', 'astro', 'bigann', 'cifar', 'instancegm', 'lastfm', 'lendb', 'uqv', 'ethz', 'movielens', 'random', 'space', 'text', 'vcseis', 'word2vec', 'geofon', 'netflix', 'nytimes', 'stead', 'txed', 'yahoomusic', 'sald', 'crawl', 'enron', 'gist', 'seismic', 'tiny5m', 'trevi', 'Music']
models=['HNSW', 'DPG', 'FLATNAV', 'GLASS-HNSW', 'VAQ', 'NNdescent', 'NSG', 'GLASS-NSG', 'PQFS', 'ITQ-LSH', 'OPQ,IMI2x1', 'RabitQ', 'LCCS-LSH', 'PQ,IMI2x2', 'E2LSH', 'DB-LSH', 'PM-LSH', 'FLANN', 'HCNNG', 'PQ', 'DISKANN', 'QALSH', 'KDTREE', 'vamana_LVQ', 'EFANNA', 'IVFPQ', 'NSSG', 'OPQ,IMI2x2', 'annoy', 'MRPT', 'vamana', 'lorann', 'scann', 'PQ,IMI2x1', 'OPQ']
list_measures=['Recall-QPS trade off','WCSR (Critical Diagram)','WCSR (Per Dataset Comparison)']
