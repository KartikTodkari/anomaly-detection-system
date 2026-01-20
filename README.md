# anomaly-detection-system
Outlier Detection (also known as Anomaly Detection) is an exciting yet challenging field, which aims to identify outlying objects that are deviant from the general data distribution. Outlier detection has been proven critical in many fields, such as credit card fraud analytics, network intrusion detection, and mechanical unit defect detection.

This repository collects:

Books & Academic Papers
Online Courses and Videos
Outlier Datasets
Open-source and Commercial Libraries/Toolkits
Key Conferences & Journals
More items will be added to the repository. Please feel free to suggest other key resources by opening an issue report, submitting a pull request, or dropping me an email @ (yzhao010@usc.edu). Enjoy reading!

BTW, you may find my [GitHub], [USC FORTIS Lab], and [Google Scholar] relevant, especially PyOD library, ADBench benchmark, and NLP-ADBench: NLP Anomaly Detection Benchmark,.

Table of Contents
1. Books & Tutorials & Benchmarks
1.1. Books
1.2. Tutorials
1.3. Benchmarks
2. Courses/Seminars/Videos
3. Toolbox & Datasets
3.1. Multivariate data outlier detection
3.2. Time series outlier detection
3.3. Graph Outlier Detection
3.4. Real-time Elasticsearch
3.5. Datasets
4. Papers
4.1. Overview & Survey Papers
4.2. Key Algorithms
4.3. Graph & Network Outlier Detection
4.4. Time Series Outlier Detection
4.5. Feature Selection in Outlier Detection
4.6. High-dimensional & Subspace Outliers
4.7. Outlier Ensembles
4.8. Outlier Detection in Evolving Data
4.9. Representation Learning in Outlier Detection
4.10. Interpretability
4.11. Outlier Detection with Neural Networks
4.12. Active Anomaly Detection
4.13. Interactive Outlier Detection
4.14. Outlier Detection in Other fields
4.15. Outlier Detection Applications
4.16. Automated Outlier Detection
4.17. Machine Learning Systems for Outlier Detection
4.18. Fairness and Bias in Outlier Detection
4.19. Isolation-based Methods
4.20. Weakly-supervised Methods
4.21. Emerging and Interesting Topics
4.22. LLM and LLM Agents for Anomaly Detection
5. Key Conferences/Workshops/Journals
5.1. Conferences & Workshops
5.2. Journals
1. Books & Tutorials & Benchmarks
1.1. Books
Outlier Analysis by Charu Aggarwal: Classical text book covering most of the outlier analysis techniques. A must-read for people in the field of outlier detection. [Preview.pdf]

Outlier Ensembles: An Introduction by Charu Aggarwal and Saket Sathe: Great intro book for ensemble learning in outlier analysis.

Data Mining: Concepts and Techniques (3rd) by Jiawei Han and Micheline Kamber and Jian Pei: Chapter 12 discusses outlier detection with many key points. [Google Search]

1.2. Tutorials
Tutorial Title	Venue	Year	Ref	Materials
Data mining for anomaly detection	PKDD	2008	[48]	[Video]
Outlier detection techniques	ACM SIGKDD	2010	[41]	[PDF]
Which Outlier Detector Should I use?	ICDM	2018	[93]	[PDF]
Deep Learning for Anomaly Detection	KDD	2020	[97]	[HTML], [Video]
Deep Learning for Anomaly Detection	WSDM	2021	[72]	[HTML]
Toward Explainable Deep Anomaly Detection	KDD	2021	[73]	[HTML]
Recent Advances in Anomaly Detection	CVPR	2023	[74]	[HTML], [Video]
Trustworthy Anomaly Detection	SDM	2024	[111]	[HTML]
1.3. Benchmarks
News: We have two new works on NLP-based and LLM-based anomaly detection:

NLP-ADBench: NLP Anomaly Detection Benchmark
AD-LLM: Benchmarking Large Language Models for Anomaly Detection
Data Types	Paper Title	Venue	Year	Ref	Materials
Time-series	Revisiting Time Series Outlier Detection: Definitions and Benchmarks	NeurIPS	2021	[45]	[PDF], [Code]
Graph	Benchmarking Node Outlier Detection on Graphs	NeurIPS	2022	[60]	[PDF], [Code]
Graph	GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection	NeurIPS	2023	[92]	[PDF], [Code]
Tabular	ADBench: Anomaly Detection Benchmark	NeurIPS	2022	[67]	[PDF], [Code]
Tabular	ADGym: Design Choices for Deep Anomaly Detection	NeurIPS	2023	[42]	[PDF], [Code]
NLP	NLP-ADBench: NLP Anomaly Detection Benchmark	Preprint	2024	[54]	[PDF], [Code]
NLP	AD-LLM: Benchmarking Large Language Models for Anomaly Detection	Preprint	2024	[103]	[PDF], [Code]
2. Courses/Seminars/Videos
Coursera Introduction to Anomaly Detection (by IBM): [See Video]

Get started with the Anomaly Detection API (by IBM): [See Website]

Practical Anomaly Detection by appliedAI Institute: [See Website], [See Video], [See GitHub]

Coursera Real-Time Cyber Threat Detection and Mitigation partly covers the topic: [See Video]

Coursera Machine Learning by Andrew Ng also partly covers the topic:

Anomaly Detection vs. Supervised Learning
Developing and Evaluating an Anomaly Detection System
Udemy Outlier Detection Algorithms in Data Mining and Data Science: [See Video]

Stanford Data Mining for Cyber Security also covers part of anomaly detection techniques: [See Video]

3. Toolbox & Datasets
[Python+LLM Agent] OpenAD: AD-AGENT is a multi-agent framework designed to automate anomaly detection across diverse data modalities, including tabular, graph, time series, and more. It integrates modular agents, model selection strategies, and configurable pipelines to support extensible and interpretable detection workflows. The framework is under active development and aims to support both academic research and practical deployment.

3.1. Multivariate Data
[Python] Python Outlier Detection (PyOD): PyOD is a comprehensive and scalable Python toolkit for detecting outlying objects in multivariate data. It contains more than 20 detection algorithms, including emerging deep learning models and outlier ensembles.

[Python, GPU] TOD: Tensor-based Outlier Detection (PyTOD): A general GPU-accelerated framework for outlier detection.

[Python] Python Streaming Anomaly Detection (PySAD): PySAD is a streaming anomaly detection framework in Python, which provides a complete set of tools for anomaly detection experiments. It currently contains more than 15 online anomaly detection algorithms and 2 different methods to integrate PyOD detectors to the streaming setting.

[Python] Scikit-learn Novelty and Outlier Detection. It supports some popular algorithms like LOF, Isolation Forest, and One-class SVM.

[Python] Scalable Unsupervised Outlier Detection (SUOD): SUOD (Scalable Unsupervised Outlier Detection) is an acceleration framework for large-scale unsupervised outlier detector training and prediction, on top of PyOD.

[Julia] OutlierDetection.jl: OutlierDetection.jl is a Julia toolkit for detecting outlying objects, also known as anomalies.

[Java] ELKI: Environment for Developing KDD-Applications Supported by Index-Structures: ELKI is an open source (AGPLv3) data mining software written in Java. The focus of ELKI is research in algorithms, with an emphasis on unsupervised methods in cluster analysis and outlier detection.

[Java] RapidMiner Anomaly Detection Extension: The Anomaly Detection Extension for RapidMiner comprises the most well know unsupervised anomaly detection algorithms, assigning individual anomaly scores to data rows of example sets. It allows you to find data, which is significantly different from the normal, without the need for the data being labeled.

[R] CRAN Task View: Anomaly Detection with R: This CRAN task view contains a list of packages that can be used for anomaly detection with R.

[R] outliers package: A collection of some tests commonly used for identifying outliers in R.

[Matlab] Anomaly Detection Toolbox - Beta: A collection of popular outlier detection algorithms in Matlab.

3.2. Time Series Outlier Detection
[Python] TODS: TODS is a full-stack automated machine learning system for outlier detection on multivariate time-series data.

[Python] skyline: Skyline is a near real time anomaly detection system.

[Python] banpei: Banpei is a Python package of the anomaly detection.

[Python] telemanom: A framework for using LSTMs to detect anomalies in multivariate time series data.

[Python] DeepADoTS: A benchmarking pipeline for anomaly detection on time series data for multiple state-of-the-art deep learning methods.

[Python] NAB: The Numenta Anomaly Benchmark: NAB is a novel benchmark for evaluating algorithms for anomaly detection in streaming, real-time applications.

[Python] CueObserve: Anomaly detection on SQL data warehouses and databases.

[Python] Chaos Genius: ML powered analytics engine for outlier/anomaly detection and root cause analysis.

[R] CRAN Task View: Anomaly Detection with R: This CRAN task view contains a list of packages that can be used for anomaly detection with R.

[R] AnomalyDetection: AnomalyDetection is an open-source R package to detect anomalies which is robust, from a statistical standpoint, in the presence of seasonality and an underlying trend.

[R] anomalize: The 'anomalize' package enables a "tidy" workflow for detecting anomalies in data.

3.3. Graph Outlier Detection
[Python] Python Graph Outlier Detection (PyGOD): PyGOD is a Python library for graph outlier detection (anomaly detection). It includes more than 10 latest graph-based detection algorithms

3.4. Real-time Elasticsearch
[Open Distro] Real Time Anomaly Detection in Open Distro for Elasticsearch by Amazon: A machine learning-based anomaly detection plugins for Open Distro for Elasticsearch. See Real Time Anomaly Detection in Open Distro for Elasticsearch.

[Python] datastream.io: An open-source framework for real-time anomaly detection using Python, Elasticsearch and Kibana.

3.5. Datasets
NLP-ADBench: NLP Anomaly Detection Benchmark and Datasets: https://github.com/USC-FORTIS/NLP-ADBench

ELKI Outlier Datasets: https://elki-project.github.io/datasets/outlier

Outlier Detection DataSets (ODDS): http://odds.cs.stonybrook.edu/#table1

Unsupervised Anomaly Detection Dataverse: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OPQMVF

Anomaly Detection Meta-Analysis Benchmarks: https://ir.library.oregonstate.edu/concern/datasets/47429f155

Skoltech Anomaly Benchmark (SKAB): https://github.com/waico/skab

4. Papers
4.1. Overview & Survey Papers
Papers are sorted by the publication year.

Paper Title	Venue	Year	Ref	Materials
A survey of outlier detection methodologies	ARTIF INTELL REV	2004	[37]	[PDF]
Anomaly detection: A survey	CSUR	2009	[19]	[PDF]
A meta-analysis of the anomaly detection problem	Preprint	2015	[29]	[PDF]
On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study	DMKD	2016	[14]	[HTML], [SLIDES]
A comparative evaluation of unsupervised anomaly detection algorithms for multivariate data	PLOS ONE	2016	[33]	[PDF]
A comparative evaluation of outlier detection algorithms: Experiments and analyses	Pattern Recognition	2018	[28]	[PDF]
Research Issues in Outlier Detection	Book Chapter	2019	[90]	[HTML]
Quantitative comparison of unsupervised anomaly detection algorithms for intrusion detection	SAC	2019	[31]	[HTML]
Progress in Outlier Detection Techniques: A Survey	IEEE Access	2019	[96]	[PDF]
Deep learning for anomaly detection: A survey	Preprint	2019	[18]	[PDF]
Anomalous Instance Detection in Deep Learning: A Survey	Tech Report	2020	[13]	[PDF]
Anomaly detection in univariate time-series: A survey on the state-of-the-art	Preprint	2020	[11]	[PDF]
Deep Learning for Anomaly Detection: A Review	CSUR	2021	[71]	[PDF]
A Comprehensive Survey on Graph Anomaly Detection with Deep Learning	TKDE	2021	[61]	[PDF]
Revisiting Time Series Outlier Detection: Definitions and Benchmarks	NeurIPS	2021	[45]	[PDF], [Code]
A Unified Survey on Anomaly, Novelty, Open-Set, and Out-of-Distribution Detection: Solutions and Future Challenges	Preprint	2021	[84]	[PDF]
Self-Supervised Anomaly Detection: A Survey and Outlook	Preprint	2022	[38]	[PDF]
Weakly supervised anomaly detection: A survey	Preprint	2023	[43]	[PDF], [PDF]
AD-LLM: Benchmarking Large Language Models for Anomaly Detection	Preprint	2024	[103]	[PDF], [Code]
Large Language Models for Anomaly and Out-of-Distribution Detection: A Survey	Preprint	2024	[102]	[PDF]
4.2. Key Algorithms
All these algorithms are available in Python Outlier Detection (PyOD).

Abbreviation	Paper Title	Venue	Year	Ref	Materials
kNN	Efficient algorithms for mining outliers from large data sets	ACM SIGMOD Record	2000	[78]	[PDF]
KNN	Fast outlier detection in high dimensional spaces	PKDD	2002	[6]	[PDF]
LOF	LOF: identifying density-based local outliers	ACM SIGMOD Record	2000	[12]	[PDF]
IForest	Isolation forest	ICDM	2008	[55]	[PDF]
OCSVM	Estimating the support of a high-dimensional distribution	Neural Computation	2001	[85]	[PDF]
AutoEncoder Ensemble	Outlier detection with autoencoder ensembles	SDM	2017	[21]	[PDF]
COPOD	COPOD: Copula-Based Outlier Detection	ICDM	2020	[50]	[PDF]
ECOD	Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions	TKDE	2022	[51]	[PDF]
4.3. Graph & Network Outlier Detection
Paper Title	Venue	Year	Ref	Materials
Graph based anomaly detection and description: a survey	DMKD	2015	[5]	[PDF]
Anomaly detection in dynamic networks: a survey	WIREs Computational Statistic	2015	[79]	[PDF]
Outlier detection in graphs: On the impact of multiple graph models	ComSIS	2019	[16]	[PDF]
A Comprehensive Survey on Graph Anomaly Detection with Deep Learning	TKDE	2021	[61]	[PDF]
4.4. Time Series Outlier Detection
Paper Title	Venue	Year	Ref	Materials
Outlier detection for temporal data: A survey	TKDE	2014	[34]	[PDF]
Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding	KDD	2018	[39]	[PDF], [Code]
Time-Series Anomaly Detection Service at Microsoft	KDD	2019	[80]	[PDF]
Revisiting Time Series Outlier Detection: Definitions and Benchmarks	NeurIPS	2021	[45]	[PDF], [Code]
Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series	ICLR	2022	[22]	[PDF], [Code]
Drift doesn't matter: dynamic decomposition with diffusion reconstruction for unstable multivariate time series anomaly detection	NeurIPS	2023	[126]	[PDF], [Code]
4.5. Feature Selection in Outlier Detection
Paper Title	Venue	Year	Ref	Materials
Unsupervised feature selection for outlier detection by modelling hierarchical value-feature couplings	ICDM	2016	[68]	[PDF]
Learning homophily couplings from non-iid data for joint feature selection and noise-resilient outlier detection	IJCAI	2017	[69]	[PDF]
4.6. High-dimensional & Subspace Outliers
Paper Title	Venue	Year	Ref	Materials
A survey on unsupervised outlier detection in high-dimensional numerical data	Stat Anal Data Min	2012	[123]	[HTML]
Learning Representations of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection	SIGKDD	2018	[70]	[PDF]
Reverse Nearest Neighbors in Unsupervised Distance-Based Outlier Detection	TKDE	2015	[77]	[PDF], [SLIDES]
Outlier detection for high-dimensional data	Biometrika	2015	[82]	[PDF]
4.7. Outlier Ensembles
Paper Title	Venue	Year	Ref	Materials
Outlier ensembles: position paper	SIGKDD Explorations	2013	[2]	[PDF]
Ensembles for unsupervised outlier detection: challenges and research questions a position paper	SIGKDD Explorations	2014	[124]	[PDF]
An Unsupervised Boosting Strategy for Outlier Detection Ensembles	PAKDD	2018	[15]	[HTML]
LSCP: Locally selective combination in parallel outlier ensembles	SDM	2019	[114]	[PDF]
Adaptive Model Pooling for Online Deep Anomaly Detection from a Complex Evolving Data Stream	KDD	2022	[108]	[PDF], [Github], [Slide]
4.8. Outlier Detection in Evolving Data
Paper Title	Venue	Year	Ref	Materials
A Survey on Anomaly detection in Evolving Data: [with Application to Forest Fire Risk Prediction]	SIGKDD Explorations	2018	[83]	[PDF]
Unsupervised real-time anomaly detection for streaming data	Neurocomputing	2017	[4]	[PDF]
Outlier Detection in Feature-Evolving Data Streams	SIGKDD	2018	[63]	[PDF], [Github]
Evaluating Real-Time Anomaly Detection Algorithms--The Numenta Anomaly Benchmark	ICMLA	2015	[47]	[PDF], [Github]
MIDAS: Microcluster-Based Detector of Anomalies in Edge Streams	AAAI	2020	[10]	[PDF], [Github]
NETS: Extremely Fast Outlier Detection from a Data Stream via Set-Based Processing	VLDB	2019	[105]	[PDF], [Github], [Slide]
Ultrafast Local Outlier Detection from a Data Stream with Stationary Region Skipping	KDD	2020	[106]	[PDF], [Github], [Slide]
Multiple Dynamic Outlier-Detection from a Data Stream by Exploiting Duality of Data and Queries	SIGMOD	2021	[107]	[PDF], [Github], [Slide]
Adaptive Model Pooling for Online Deep Anomaly Detection from a Complex Evolving Data Stream	KDD	2022	[108]	[PDF], [Github], [Slide]
4.9. Representation Learning in Outlier Detection
Paper Title	Venue	Year	Ref	Materials
Learning Representations of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection	SIGKDD	2018	[70]	[PDF]
Learning representations for outlier detection on a budget	Preprint	2015	[65]	[PDF]
XGBOD: improving supervised outlier detection with unsupervised representation learning	IJCNN	2018	[113]	[PDF]
4.10. Interpretability
Paper Title	Venue	Year	Ref	Materials
Explaining Anomalies in Groups with Characterizing Subspace Rules	DMKD	2018	[62]	[PDF]
Beyond Outlier Detection: LookOut for Pictorial Explanation	ECML-PKDD	2018	[66]	[PDF]
Contextual outlier interpretation	IJCAI	2018	[57]	[PDF]
Mining multidimensional contextual outliers from categorical relational data	IDA	2015	[91]	[PDF]
Discriminative features for identifying and interpreting outliers	ICDE	2014	[23]	[PDF]
Sequential Feature Explanations for Anomaly Detection	TKDD	2019	[88]	[HTML]
A Survey on Explainable Anomaly Detection	TKDD	2023	[52]	[HTML]
Explainable Contextual Anomaly Detection Using Quantile Regression Forests	DMKD	2023	[53]	[HTML]
Beyond Outlier Detection: Outlier Interpretation by Attention-Guided Triplet Deviation Network	WWW	2021	[99]	[PDF]
4.11. Outlier Detection with Neural Networks
Paper Title	Venue	Year	Ref	Materials
Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding	KDD	2018	[39]	[PDF], [Code]
MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks	ICANN	2019	[49]	[PDF], [Code]
Generative Adversarial Active Learning for Unsupervised Outlier Detection	TKDE	2019	[58]	[PDF], [Code]
Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection	ICLR	2018	[125]	[PDF], [Code]
Deep Anomaly Detection with Outlier Exposure	ICLR	2019	[36]	[PDF], [Code]
Unsupervised Anomaly Detection With LSTM Neural Networks	TNNLS	2019	[30]	[PDF], [IEEE],
Effective End-to-end Unsupervised Outlier Detection via Inlier Priority of Discriminative Network	NeurIPS	2019	[95]	[PDF] [Code]
Fascinating Supervisory Signals and Where to Find Them: Deep Anomaly Detection with Scale Learning	ICML	2023	[101]	[PDF], [Code]
4.12. Active Anomaly Detection
Paper Title	Venue	Year	Ref	Materials
Active learning for anomaly and rare-category detection	NeurIPS	2005	[76]	[PDF]
Outlier detection by active learning	SIGKDD	2006	[1]	[PDF]
Active Anomaly Detection via Ensembles: Insights, Algorithms, and Interpretability	Preprint	2019	[24]	[PDF]
Meta-AAD: Active Anomaly Detection with Deep Reinforcement Learning	ICDM	2020	[112]	[PDF]
A3: Activation Anomaly Analysis	ECML-PKDD	2020	[89]	[PDF], [Code]
4.13. Interactive Outlier Detection
Paper Title	Venue	Year	Ref	Materials
Learning On-the-Job to Re-rank Anomalies from Top-1 Feedback	SDM	2019	[46]	[PDF]
Interactive anomaly detection on attributed networks	WSDM	2019	[26]	[PDF]
eX2: a framework for interactive anomaly detection	IUI Workshop	2019	[7]	[PDF]
Tripartite Active Learning for Interactive Anomaly Discovery	IEEE Access	2019	[122]	[PDF]
4.14. Outlier Detection in Other fields
Field	Paper Title	Venue	Year	Ref	Materials
Text	Outlier detection for text data	SDM	2017	[40]	[PDF]
4.15. Outlier Detection Applications
Field	Paper Title	Venue	Year	Ref	Materials
Security	A survey of distance and similarity measures used within network intrusion anomaly detection	IEEE Commun. Surv. Tutor.	2015	[98]	[PDF]
Security	Anomaly-based network intrusion detection: Techniques, systems and challenges	Computers & Security	2009	[32]	[PDF]
Finance	A survey of anomaly detection techniques in financial domain	Future Gener Comput Syst	2016	[3]	[PDF]
Traffic	Outlier Detection in Urban Traffic Data	WIMS	2018	[27]	[PDF]
Social Media	A survey on social media anomaly detection	SIGKDD Explorations	2016	[110]	[PDF]
Social Media	GLAD: group anomaly detection in social media analysis	TKDD	2015	[109]	[PDF]
Machine Failure	Detecting the Onset of Machine Failure Using Anomaly Detection Methods	DAWAK	2019	[81]	[PDF]
Video Surveillance	AnomalyNet: An anomaly detection network for video surveillance	TIFS	2019	[120]	[IEEE], Code
4.16. Automated Outlier Detection
Paper Title	Venue	Year	Ref	Materials
AutoML: state of the art with a focus on anomaly detection, challenges, and research directions	Int J Data Sci Anal	2022	[8]	[PDF]
AutoOD: Automated Outlier Detection via Curiosity-guided Search and Self-imitation Learning	ICDE	2020	[59]	[PDF]
Automatic Unsupervised Outlier Model Selection	NeurIPS	2021	[116]	[PDF], [Code]
4.17. Machine Learning Systems for Outlier Detection
This section summarizes a list of systems for outlier detection, which may overlap with the section of tools and libraries.

Paper Title	Venue	Year	Ref	Materials
PyOD: A Python Toolbox for Scalable Outlier Detection	JMLR	2019	[115]	[PDF], [Code]
SUOD: Accelerating Large-Scale Unsupervised Heterogeneous Outlier Detection	MLSys	2021	[117]	[PDF], [Code]
TOD: Tensor-based Outlier Detection	Preprint	2021	[118]	[PDF], [Code]
4.18. Fairness and Bias in Outlier Detection
Paper Title	Venue	Year	Ref	Materials
A Framework for Determining the Fairness of Outlier Detection	ECAI	2020	[25]	[PDF]
FAIROD: Fairness-aware Outlier Detection	AIES	2021	[87]	[PDF]
4.19. Isolation-Based Methods
Paper Title	Venue	Year	Ref	Materials
Isolation forest	ICDM	2008	[55]	[PDF]
Isolation‐based anomaly detection using nearest‐neighbor ensembles	Computational Intelligence	2018	[9]	[PDF], [Code]
Extended Isolation Forest	TKDE	2019	[35]	[PDF], [Code]
Isolation Distributional Kernel: A New Tool for Kernel based Anomaly Detection	KDD	2020	[94]	[PDF], [Code]
Deep Isolation Forest for Anomaly Detection	TKDE	2023	[100]	[PDF], [Code]
4.20. Weakly-Supervised Methods
Paper Title	Venue	Year	Ref	Materials
XGBOD: improving supervised outlier detection with unsupervised representation learning	IJCNN	2018	[113]	[PDF]
Feature Encoding With Autoencoders for Weakly Supervised Anomaly Detection	TNNLS	2021	[121]	[PDF], [Code]
4.21. Emerging and Interesting Topics
Paper Title	Venue	Year	Ref	Materials
Clustering with Outlier Removal	TKDE	2019	[56]	[PDF]
Real-World Anomaly Detection by using Digital Twin Systems and Weakly-Supervised Learning	IEEE Trans. Ind. Informat.	2020	[17]	[PDF]
SSD: A Unified Framework for Self-Supervised Outlier Detection	ICLR	2021	[86]	[PDF], [Code]
AD-LLM: Benchmarking Large Language Models for Anomaly Detection	Preprint	2024	[103]	[PDF], [Code]
4.22. LLM and LLM Agents for Anomaly Detection
Paper Title	Venue	Year	Ref	Materials
AD-LLM: Benchmarking Large Language Models for Anomaly Detection	ACL 2025 Findings	2024	[103]	[PDF], [Code]
NLP-ADBench: NLP Anomaly Detection Benchmark	EMNLP 2025 Findings	2024	[54]	[PDF], [Code]
AD-AGENT: A Multi-agent Framework for End-to-end Anomaly Detection	Findings of IJCNLP-AACL	2025	[104]	[PDF], [Code]
LogSAD: Training-free Anomaly Detection with Vision & Language Foundation Models	CVPR 2025	2025	[119]	[PDF], [Code]
MMAD: A Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection	ICLR 2025	2025	[44]	[PDF], [Code]
Delving into Large Language Models for Effective Time-Series Anomaly Detection	NeurIPS 2025	2025	[75]	[PDF], [Code]
5. Key Conferences/Workshops/Journals
5.1. Conferences & Workshops
Key data mining conference deadlines, historical acceptance rates, and more can be found data-mining-conferences.

ACM International Conference on Knowledge Discovery and Data Mining (SIGKDD). Note: SIGKDD usually has an Outlier Detection Workshop (ODD), see ODD 2021.

ACM International Conference on Management of Data (SIGMOD)

The Web Conference (WWW)

IEEE International Conference on Data Mining (ICDM)

SIAM International Conference on Data Mining (SDM)

IEEE International Conference on Data Engineering (ICDE)

ACM InternationalConference on Information and Knowledge Management (CIKM)

ACM International Conference on Web Search and Data Mining (WSDM)

The European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)

The Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)

5.2. Journals
ACM Transactions on Knowledge Discovery from Data (TKDD)

IEEE Transactions on Knowledge and Data Engineering (TKDE)

ACM SIGKDD Explorations Newsletter

Data Mining and Knowledge Discovery

Knowledge and Information Systems (KAIS)

References
[1]	Abe, N., Zadrozny, B. and Langford, J., 2006, August. Outlier detection by active learning. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 504-509, ACM.
[2]	Aggarwal, C.C., 2013. Outlier ensembles: position paper. ACM SIGKDD Explorations Newsletter, 14(2), pp.49-58.
[3]	Ahmed, M., Mahmood, A.N. and Islam, M.R., 2016. A survey of anomaly detection techniques in financial domain. Future Generation Computer Systems, 55, pp.278-288.
[4]	Ahmad, S., Lavin, A., Purdy, S. and Agha, Z., 2017. Unsupervised real-time anomaly detection for streaming data. Neurocomputing, 262, pp.134-147.
[5]	Akoglu, L., Tong, H. and Koutra, D., 2015. Graph based anomaly detection and description: a survey. Data Mining and Knowledge Discovery, 29(3), pp.626-688.
[6]	Angiulli, F. and Pizzuti, C., 2002, August. Fast outlier detection in high dimensional spaces. In European Conference on Principles of Data Mining and Knowledge Discovery, pp. 15-27.
[7]	Arnaldo, I., Veeramachaneni, K. and Lam, M., 2019. ex2: a framework for interactive anomaly detection. In ACM IUI Workshop on Exploratory Search and Interactive Data Analytics (ESIDA).
[8]	Bahri, M., Salutari, F., Putina, A. et al. AutoML: state of the art with a focus on anomaly detection, challenges, and research directions. International Journal of Data Science and Analytics (2022).
[9]	Bandaragoda, Tharindu R., Kai Ming Ting, David Albrecht, Fei Tony Liu, Ye Zhu, and Jonathan R. Wells. "Isolation‐based anomaly detection using nearest‐neighbor ensembles." Computational Intelligence 34, no. 4 (2018): 968-998.
[10]	Bhatia, S., Hooi, B., Yoon, M., Shin, K. and Faloutsos. C., 2020. MIDAS: Microcluster-Based Detector of Anomalies in Edge Streams. In AAAI Conference on Artificial Intelligence (AAAI).
[11]	Braei, M. and Wagner, S., 2020. Anomaly detection in univariate time-series: A survey on the state-of-the-art. arXiv preprint arXiv:2004.00433.
[12]	Breunig, M.M., Kriegel, H.P., Ng, R.T. and Sander, J., 2000, May. LOF: identifying density-based local outliers. ACM SIGMOD Record, 29(2), pp. 93-104.
[13]	Bulusu, S., Kailkhura, B., Li, B., Varshney, P. and Song, D., 2020. Anomalous instance detection in deep learning: A survey (No. LLNL-CONF-808677). Lawrence Livermore National Lab.(LLNL), Livermore, CA (United States).
[14]	Campos, G.O., Zimek, A., Sander, J., Campello, R.J., Micenková, B., Schubert, E., Assent, I. and Houle, M.E., 2016. On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study. Data Mining and Knowledge Discovery, 30(4), pp.891-927.
[15]	Campos, G.O., Zimek, A. and Meira, W., 2018, June. An Unsupervised Boosting Strategy for Outlier Detection Ensembles. In Pacific-Asia Conference on Knowledge Discovery and Data Mining (pp. 564-576). Springer, Cham.
[16]	Campos, G.O., Moreira, E., Meira Jr, W. and Zimek, A., 2019. Outlier Detection in Graphs: A Study on the Impact of Multiple Graph Models. Computer Science & Information Systems, 16(2).
[17]	Castellani, A., Schmitt, S., Squartini, S., 2020. Real-World Anomaly Detection by using Digital Twin Systems and Weakly-Supervised Learning. In IEEE Transactions on Industrial Informatics.
[18]	Chalapathy, R. and Chawla, S., 2019. Deep learning for anomaly detection: A survey. arXiv preprint arXiv:1901.03407.
[19]	Chandola, V., Banerjee, A. and Kumar, V., 2009. Anomaly detection: A survey. ACM computing surveys , 41(3), p.15.
[20]	Chawla, S. and Chandola, V., 2011, Anomaly Detection: A Tutorial. Tutorial at ICDM 2011.
[21]	Chen, J., Sathe, S., Aggarwal, C. and Turaga, D., 2017, June. Outlier detection with autoencoder ensembles. SIAM International Conference on Data Mining, pp. 90-98. Society for Industrial and Applied Mathematics.
[22]	Dai, E. and Chen, J., 2022. Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series. International Conference on Learning Representations (ICLR).
[23]	Dang, X.H., Assent, I., Ng, R.T., Zimek, A. and Schubert, E., 2014, March. Discriminative features for identifying and interpreting outliers. In International Conference on Data Engineering (ICDE). IEEE.
[24]	Das, S., Islam, M.R., Jayakodi, N.K. and Doppa, J.R., 2019. Active Anomaly Detection via Ensembles: Insights, Algorithms, and Interpretability. arXiv preprint arXiv:1901.08930.
[25]	Davidson, I. and Ravi, S.S., 2020. A framework for determining the fairness of outlier detection. In Proceedings of the 24th European Conference on Artificial Intelligence (ECAI2020) (Vol. 2029).
[26]	Ding, K., Li, J. and Liu, H., 2019, January. Interactive anomaly detection on attributed networks. In Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining, pp. 357-365. ACM.
[27]	Djenouri, Y. and Zimek, A., 2018, June. Outlier detection in urban traffic data. In Proceedings of the 8th International Conference on Web Intelligence, Mining and Semantics. ACM.
[28]	Domingues, R., Filippone, M., Michiardi, P. and Zouaoui, J., 2018. A comparative evaluation of outlier detection algorithms: Experiments and analyses. Pattern Recognition, 74, pp.406-421.
[29]	Emmott, A., Das, S., Dietterich, T., Fern, A. and Wong, W.K., 2015. A meta-analysis of the anomaly detection problem. arXiv preprint arXiv:1503.01158.
[30]	Ergen, T. and Kozat, S.S., 2019. Unsupervised Anomaly Detection With LSTM Neural Networks. IEEE transactions on neural networks and learning systems.
[31]	Falcão, F., Zoppi, T., Silva, C.B.V., Santos, A., Fonseca, B., Ceccarelli, A. and Bondavalli, A., 2019, April. Quantitative comparison of unsupervised anomaly detection algorithms for intrusion detection. In Proceedings of the 34th ACM/SIGAPP Symposium on Applied Computing, (pp. 318-327). ACM.
[32]	Garcia-Teodoro, P., Diaz-Verdejo, J., Maciá-Fernández, G. and Vázquez, E., 2009. Anomaly-based network intrusion detection: Techniques, systems and challenges. Computers & Security, 28(1-2), pp.18-28.
[33]	Goldstein, M. and Uchida, S., 2016. A comparative evaluation of unsupervised anomaly detection algorithms for multivariate data. PloS one, 11(4), p.e0152173.
[34]	Gupta, M., Gao, J., Aggarwal, C.C. and Han, J., 2014. Outlier detection for temporal data: A survey. IEEE Transactions on Knowledge and Data Engineering, 26(9), pp.2250-2267.
[35]	Hariri, S., Kind, M.C. and Brunner, R.J., 2019. Extended Isolation Forest. IEEE Transactions on Knowledge and Data Engineering.
[36]	Hendrycks, D., Mazeika, M. and Dietterich, T.G., 2019. Deep Anomaly Detection with Outlier Exposure. International Conference on Learning Representations (ICLR).
[37]	Hodge, V. and Austin, J., 2004. A survey of outlier detection methodologies. Artificial intelligence review, 22(2), pp.85-126.
[38]	Hojjati, H., Ho, T.K.K. and Armanfard, N., 2022. Self-Supervised Anomaly Detection: A Survey and Outlook. arXiv preprint arXiv:2205.05173.
[39]	(1, 2) Hundman, K., Constantinou, V., Laporte, C., Colwell, I. and Soderstrom, T., 2018, July. Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, (pp. 387-395). ACM.
[40]	Kannan, R., Woo, H., Aggarwal, C.C. and Park, H., 2017, June. Outlier detection for text data. In Proceedings of the 2017 SIAM International Conference on Data Mining, pp. 489-497. Society for Industrial and Applied Mathematics.
[41]	Kriegel, H.P., Kröger, P. and Zimek, A., 2010. Outlier detection techniques. Tutorial at ACM SIGKDD 2010.
[42]	Jiang, M., Hou, C., Zheng, A., Han, S., Huang, H., Wen, Q., Hu, X. and Zhao, Y., 2023. ADGym: Design Choices for Deep Anomaly Detection. NeurIPS, Datasets and Benchmarks Track.
[43]	Jiang, M., Hou, C., Zheng, A., Hu, X., Han, S., Huang, H., He, X., Yu, P.S. and Zhao, Y., 2023. Weakly supervised anomaly detection: A survey. arXiv preprint arXiv:2302.04549.
[44]	Jiang, X., Li, J., Deng, H., Liu, Y., Gao, B., Zhou, Y., Li, J., Wang, C. and Zheng, F., 2025. MMAD: A Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection. In ICLR 2025.
[45]	(1, 2, 3) Lai, K.H., Zha, D., Xu, J., Zhao, Y., Wang, G. and Hu, X., 2021. Revisiting Time Series Outlier Detection: Definitions and Benchmarks. NeurIPS, Datasets and Benchmarks Track.
[46]	Lamba, H. and Akoglu, L., 2019, May. Learning On-the-Job to Re-rank Anomalies from Top-1 Feedback. In Proceedings of the 2019 SIAM International Conference on Data Mining (SDM), pp. 612-620. Society for Industrial and Applied Mathematics.
[47]	Lavin, A. and Ahmad, S., 2015, December. Evaluating Real-Time Anomaly Detection Algorithms--The Numenta Anomaly Benchmark. In 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA) (pp. 38-44). IEEE.
[48]	Lazarevic, A., Banerjee, A., Chandola, V., Kumar, V. and Srivastava, J., 2008, September. Data mining for anomaly detection. Tutorial at ECML PKDD 2008.
[49]	Li, D., Chen, D., Jin, B., Shi, L., Goh, J. and Ng, S.K., 2019, September. MAD-GAN: Multivariate anomaly detection for time series data with generative adversarial networks. In International Conference on Artificial Neural Networks (pp. 703-716). Springer, Cham.
[50]	Li, Z., Zhao, Y., Botta, N., Ionescu, C. and Hu, X. COPOD: Copula-Based Outlier Detection. IEEE International Conference on Data Mining (ICDM), 2020.
[51]	Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C. and Chen, H. G. ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions. IEEE Transactions on Knowledge and Data Engineering (TKDE), 2022.
[52]	Li, Z., Zhu, Y. and Van Leeuwen, M., 2023. A survey on explainable anomaly detection. ACM Transactions on Knowledge Discovery from Data, 18(1), pp.1-54.
[53]	Li, Z. and Van Leeuwen, M., 2023. Explainable contextual anomaly detection using quantile regression forests. Data Mining and Knowledge Discovery, 37(6), pp.2517-2563.
[54]	(1, 2) Li, Y., Li, J., Xiao, Z., Yang, T., Nian, Y., Hu, X. and Zhao, Y. "NLP-ADBench: NLP Anomaly Detection Benchmark," arXiv preprint arXiv:2412.04784.
[55]	(1, 2) Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008, December. Isolation forest. In International Conference on Data Mining, pp. 413-422. IEEE.
[56]	Liu, H., Li, J., Wu, Y. and Fu, Y., 2019. Clustering with outlier removal. IEEE transactions on knowledge and data engineering.
[57]	Liu, N., Shin, D. and Hu, X., 2017. Contextual outlier interpretation. In International Joint Conference on Artificial Intelligence (IJCAI-18), pp.2461-2467.
[58]	Liu, Y., Li, Z., Zhou, C., Jiang, Y., Sun, J., Wang, M. and He, X., 2019. Generative Adversarial Active Learning for Unsupervised Outlier Detection. IEEE transactions on knowledge and data engineering.
[59]	Li, Y., Chen, Z., Zha, D., Zhou, K., Jin, H., Chen, H. and Hu, X., 2020. AutoOD: Automated Outlier Detection via Curiosity-guided Search and Self-imitation Learning. ICDE.
[60]	Liu, K., Dou, Y., Zhao, Y., Ding, X., Hu, X., Zhang, R., Ding, K., Chen, C., Peng, H., Shu, K., Sun, L., Li, J., Chen, G.H., Jia, Z., and Yu, P.S. 2022. Benchmarking Node Outlier Detection on Graphs. arXiv preprint arXiv:2206.10071.
[61]	(1, 2) Ma, X., Wu, J., Xue, S., Yang, J., Zhou, C., Sheng, Q.Z., Xiong, H. and Akoglu, L., 2021. A comprehensive survey on graph anomaly detection with deep learning. IEEE Transactions on Knowledge and Data Engineering.
[62]	Macha, M. and Akoglu, L., 2018. Explaining anomalies in groups with characterizing subspace rules. Data Mining and Knowledge Discovery, 32(5), pp.1444-1480.
[63]	Manzoor, E., Lamba, H. and Akoglu, L. Outlier Detection in Feature-Evolving Data Streams. In 24th ACM SIGKDD International Conference on Knowledge Discovery and Data mining (KDD). 2018.
[64]	Mendiratta, B.V., 2017. Anomaly Detection in Networks. Tutorial at ACM SIGKDD 2017.
[65]	Micenková, B., McWilliams, B. and Assent, I., 2015. Learning representations for outlier detection on a budget. arXiv preprint arXiv:1507.08104.
[66]	Gupta, N., Eswaran, D., Shah, N., Akoglu, L. and Faloutsos, C., Beyond Outlier Detection: LookOut for Pictorial Explanation. ECML PKDD 2018.
[67]	Han, S., Hu, X., Huang, H., Jiang, M. and Zhao, Y., 2022. ADBench: Anomaly Detection Benchmark. arXiv preprint arXiv:2206.09426.
[68]	Pang, G., Cao, L., Chen, L. and Liu, H., 2016, December. Unsupervised feature selection for outlier detection by modelling hierarchical value-feature couplings. In Data Mining (ICDM), 2016 IEEE 16th International Conference on (pp. 410-419). IEEE.
[69]	Pang, G., Cao, L., Chen, L. and Liu, H., 2017, August. Learning homophily couplings from non-iid data for joint feature selection and noise-resilient outlier detection. In Proceedings of the 26th International Joint Conference on Artificial Intelligence (pp. 2585-2591). AAAI Press.
[70]	(1, 2) Pang, G., Cao, L., Chen, L. and Liu, H., 2018. Learning Representations of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection. In 24th ACM SIGKDD International Conference on Knowledge Discovery and Data mining (KDD). 2018.
[71]	Pang, G., Shen, C., Cao, L. and Hengel, A.V.D., 2021. Deep Learning for Anomaly Detection: A Review. ACM Computing Surveys (CSUR), 54(2), pp.1-38.
[72]	Pang, G., Cao, L. and Aggarwal, C., 2021. Deep Learning for Anomaly Detection. Tutorial at WSDM 2021.
[73]	Pang, G. and Aggarwal, C., 2021, August. Toward explainable deep anomaly detection. In KDD (pp. 4056-4057).
[74]	Guansong Pang, Joey Tianyi Zhou, Radu Tudor Ionescu, Yu Tian, and Kihyuk Sohn. "Recent Advances in Anomaly Detection". In: CVPR'23. Vancouver, Canada.
[75]	Park, J., Jung, K., Lee, D., Lee, H., Gwak, D., Park, C., Choo, J. and Cho, J., 2025. Delving into Large Language Models for Effective Time-Series Anomaly Detection. In NeurIPS 2025.
[76]	Pelleg, D. and Moore, A.W., 2005. Active learning for anomaly and rare-category detection. In Advances in neural information processing systems, pp. 1073-1080.
[77]	Radovanović, M., Nanopoulos, A. and Ivanović, M., 2015. Reverse nearest neighbors in unsupervised distance-based outlier detection. IEEE transactions on knowledge and data engineering, 27(5), pp.1369-1382.
[78]	Ramaswamy, S., Rastogi, R. and Shim, K., 2000, May. Efficient algorithms for mining outliers from large data sets. ACM SIGMOD Record, 29(2), pp. 427-438.
[79]	Ranshous, S., Shen, S., Koutra, D., Harenberg, S., Faloutsos, C. and Samatova, N.F., 2015. Anomaly detection in dynamic networks: a survey. Wiley Interdisciplinary Reviews: Computational Statistics, 7(3), pp.223-247.
[80]	Ren, H., Xu, B., Wang, Y., Yi, C., Huang, C., Kou, X., Xing, T., Yang, M., Tong, J. and Zhang, Q., 2019. Time-Series Anomaly Detection Service at Microsoft. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM.
[81]	Riazi, M., Zaiane, O., Takeuchi, T., Maltais, A., Günther, J. and Lipsett, M., Detecting the Onset of Machine Failure Using Anomaly Detection Methods.
[82]	Ro, K., Zou, C., Wang, Z. and Yin, G., 2015. Outlier detection for high-dimensional data. Biometrika, 102(3), pp.589-599.
[83]	Salehi, Mahsa & Rashidi, Lida. (2018). A Survey on Anomaly detection in Evolving Data: [with Application to Forest Fire Risk Prediction]. ACM SIGKDD Explorations Newsletter. 20. 13-23.
[84]	Salehi, M., Mirzaei, H., Hendrycks, D., Li, Y., Rohban, M.H., Sabokrou, M., 2021. A Unified Survey on Anomaly, Novelty, Open-Set, and Out-of-Distribution Detection: Solutions and Future Challenges. arXiv preprint arXiv:2110.14051.
[85]	Schölkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J. and Williamson, R.C., 2001. Estimating the support of a high-dimensional distribution. Neural Computation, 13(7), pp.1443-1471.
[86]	Sehwag, V., Chiang, M., Mittal, P., 2021. SSD: A Unified Framework for Self-Supervised Outlier Detection. International Conference on Learning Representations (ICLR).
[87]	Shekhar, S., Shah, N. and Akoglu, L., 2021. FAIROD: Fairness-aware Outlier Detection. AAAI/ACM Conference on AI, Ethics, and Society (AIES).
[88]	Siddiqui, M.A., Fern, A., Dietterich, T.G. and Wong, W.K., 2019. Sequential Feature Explanations for Anomaly Detection. ACM Transactions on Knowledge Discovery from Data (TKDD), 13(1), p.1.
[89]	Sperl, P., Schulze, J.-P., and Böttinger, K., 2021. Activation Anomaly Analysis. European Conference on Machine Learning and Data Mining (ECML-PKDD) 2020.
[90]	Suri, N.R. and Athithan, G., 2019. Research Issues in Outlier Detection. In Outlier Detection: Techniques and Applications, pp. 29-51. Springer, Cham.
[91]	Tang, G., Pei, J., Bailey, J. and Dong, G., 2015. Mining multidimensional contextual outliers from categorical relational data. Intelligent Data Analysis, 19(5), pp.1171-1192.
[92]	Tang, J., Hua, F., Gao, Z., Zhao, P. and Li, J., 2023. GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection. NeurIPS, Datasets and Benchmarks Track.
[93]	Ting, KM., Aryal, S. and Washio, T., 2018, Which Anomaly Detector should I use? Tutorial at ICDM 2018.
[94]	Ting, Kai Ming, Bi-Cun Xu, Takashi Washio, and Zhi-Hua Zhou. "Isolation Distributional Kernel: A New Tool for Kernel based Anomaly Detection." In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 198-206. 2020.
[95]	Wang, S., Zeng, Y., Liu, X., Zhu, E., Yin, J., Xu, C. and Kloft, M., 2019. Effective End-to-end Unsupervised Outlier Detection via Inlier Priority of Discriminative Network. In 33rd Conference on Neural Information Processing Systems.
[96]	Wang, H., Bah, M.J. and Hammad, M., 2019. Progress in Outlier Detection Techniques: A Survey. IEEE Access, 7, pp.107964-108000.
[97]	Wang, R., Nie, K., Chang, Y. J., Gong, X., Wang, T., Yang, Y., Long, B., 2020. Deep Learning for Anomaly Detection. Tutorial at KDD 2020.
[98]	Weller-Fahy, D.J., Borghetti, B.J. and Sodemann, A.A., 2015. A survey of distance and similarity measures used within network intrusion anomaly detection. IEEE Communications Surveys & Tutorials, 17(1), pp.70-91.
[99]	Xu, H., Wang, Y., Jian, S., Huang, Z., Wang, Y., Liu, N. and Li, F., 2021, April. Beyond Outlier Detection: Outlier Interpretation by Attention-Guided Triplet Deviation Network. In Proceedings of the Web Conference 2021 (pp. 1328-1339).
[100]	Xu, H., Pang, G., Wang, Y., Wang, Y., 2023. Deep Isolation Forest for Anomaly Detection. IEEE Transactions on Knowledge and Data Engineering.
[101]	Xu, H., Wang, Y., Wei, J., Jian, S., Li, Y., Liu, N., 2023. Fascinating Supervisory Signals and Where to Find Them: Deep Anomaly Detection with Scale Learning. International Conference on Machine Learning (ICML).
[102]	Xu, R. and Ding, K., 2024. Large language models for anomaly and out-of-distribution detection: A survey. arXiv preprint arXiv:2409.01980.
[103]	(1, 2, 3, 4) Yang, T., Nian, Y., Li, S., Xu, R., Li, Y., Li, J., Xiao, Z., Hu, X., Rossi, R., Ding, K., Hu, X. and Zhao, Y. "AD-LLM: Benchmarking Large Language Models for Anomaly Detection." Findings of ACL, 2025.
[104]	Yang, T., Liu, J., Siu, W., Wang, J., Qian, Z., Song, C., Cheng, C., Hu, X., and Zhao, Y. "AD-AGENT: A Multi-agent Framework for End-to-end Anomaly Detection." Findings of IJCNLP-AACL, 2025.
[105]	Yoon, S., Lee, J. G., & Lee, B. S., 2019. NETS: extremely fast outlier detection from a data stream via set-based processing. Proceedings of the VLDB Endowment, 12(11), 1303-1315.
[106]	Yoon, S., Lee, J. G., & Lee, B. S., 2020. Ultrafast local outlier detection from a data stream with stationary region skipping. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 1181-1191)
[107]	Yoon, S., Shin, Y., Lee, J. G., & Lee, B. S. (2021, June). Multiple dynamic outlier-detection from a data stream by exploiting duality of data and queries. In Proceedings of the 2021 International Conference on Management of Data (SIGMOD).
[108]	(1, 2) Yoon, S., Lee, Y., Lee, J.G. and Lee, B.S., 2022, August. Adaptive Model Pooling for Online Deep Anomaly Detection from a Complex Evolving Data Stream. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 2347-2357).
[109]	Yu, R., He, X. and Liu, Y., 2015. GLAD: group anomaly detection in social media analysis. ACM Transactions on Knowledge Discovery from Data (TKDD), 10(2), p.18.
[110]	Yu, R., Qiu, H., Wen, Z., Lin, C. and Liu, Y., 2016. A survey on social media anomaly detection. ACM SIGKDD Explorations Newsletter, 18(1), pp.1-14.
[111]	Yuan, S., Xu, D. and Wu, X., 2024 Trustworthy Anomaly Detection. Tutorial at SDM 2024.
[112]	Zha, D., Lai, K.H., Wan, M. and Hu, X., 2020. Meta-AAD: Active Anomaly Detection with Deep Reinforcement Learning. ICDM.
[113]	(1, 2) Zhao, Y. and Hryniewicki, M.K., 2018, July. XGBOD: improving supervised outlier detection with unsupervised representation learning. In 2018 International Joint Conference on Neural Networks (IJCNN). IEEE.
[114]	Zhao, Y., Nasrullah, Z., Hryniewicki, M.K. and Li, Z., 2019, May. LSCP: Locally selective combination in parallel outlier ensembles. In Proceedings of the 2019 SIAM International Conference on Data Mining (SDM), pp. 585-593. Society for Industrial and Applied Mathematics.
[115]	Zhao, Y., Nasrullah, Z. and Li, Z., PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of Machine Learning Research, 20, pp.1-7.
[116]	Zhao, Y., Rossi, R.A. and Akoglu, L., 2021. Automatic Unsupervised Outlier Model Selection. Advances in Neural Information Processing Systems.
[117]	Zhao, Y., Hu, X., Cheng, C., Wang, C., Wan, C., Wang, W., Yang, J., Bai, H., Li, Z., Xiao, C. and Wang, Y., 2021. SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection. Proceedings of Machine Learning and Systems (MLSys).
[118]	Zhao, Y., Chen, G.H. and Jia, Z., 2021. TOD: Tensor-based Outlier Detection. arXiv preprint arXiv:2110.14007.
[119]	Zhang, J., Wang, G., Jin, Y. and Huang, D., 2025. Towards Training-free Anomaly Detection with Vision and Language Foundation Models. In CVPR 2025.
[120]	Zhou, J.T., Du, J., Zhu, H., Peng, X., Liu, Y. and Goh, R.S.M., 2019. AnomalyNet: An anomaly detection network for video surveillance. IEEE Transactions on Information Forensics and Security.
[121]	Zhou, Y., Song, X., Zhang, Y., Liu, F., Zhu, C., & Liu, L. (2021). Feature encoding with autoencoders for weakly supervised anomaly detection. IEEE Transactions on Neural Networks and Learning Systems, 33(6), 2454-2465.
[122]	Zhu, Y. and Yang, K., 2019. Tripartite Active Learning for Interactive Anomaly Discovery. IEEE Access.
[123]	Zimek, A., Schubert, E. and Kriegel, H.P., 2012. A survey on unsupervised outlier detection in high‐dimensional numerical data. Statistical Analysis and Data Mining: The ASA Data Science Journal, 5(5), pp.363-387.
[124]	Zimek, A., Campello, R.J. and Sander, J., 2014. Ensembles for unsupervised outlier detection: challenges and research questions a position paper. ACM Sigkdd Explorations Newsletter, 15(1), pp.11-22.
[125]	Zong, B., Song, Q., Min, M.R., Cheng, W., Lumezanu, C., Cho, D. and Chen, H., 2018. Deep autoencoding gaussian mixture model for unsupervised anomaly detection. International Conference on Learning Representations (ICLR).
[126]	Wang, C., Zhuang, Z., Qi, Q., Wang, J., Wang, X., Sun, H., & Liao, J. (2023). Drift doesn't matter: dynamic decomposition with diffusion reconstruction for unstable multivariate time series anomaly detection. Advances in Neural Information Processing Systems, 36.
