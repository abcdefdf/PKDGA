# PKDGA
PKDGA is an adversarial domain generation algorithm that uses a small amount of detector information for reinforcement learning training.

# Installation and Usage Hints
PKDGA was tested and used on Ubuntu 16.04 and Win10 using Anaconda Python and Pycharm Environments.

# Introduction
## PKDGA
generator.py: generator model.  
rollout.py: policy gradient, to make the token sequence Y resist against target detector D, we maximize its expected reward.  
gen_model.py: train the model.  exp1.py: the first experimental code in the paper.  
gen_sample.py: generate domain names by loading the trained model.  
exp1.py: the first experimental code in the paper.  
exp2.py: the second experimental code in the paper.  
file_check.py: other functions that need to be called.   
## RF
DGA detector, it extracts 21 lightweight DGA features, including structural, linguistic and statistical features, and employs a supervised model (Random Forest) for classification.
## bilstm,lstm,textcnn
DGA detector, directly leverage neural networks to distinguish AGDs from benign ones without extracting features manually.
## graph
DGA detector, graph is able to detect dictionary-based AGDs that are generated by concatenating words from a predefined dictionary. It constructs a graph describing the relationships between the largest common substrings in a domain and the graph with average degrees larger than a threshold is considered as malicious. In our implementation, the substrings that repeat more than three times are considered as basic words, and the degree threshold is set to 3.
## statics
DGA detector, argues that AGDs follow different distributions from the legitimate domain names. A domain is identified as benign if it has a negligible distance from the legitimate domains. In particular, it measures the distance using Kullback-Leibler divergence, Jaccard index and Edit distance.
## khaos
DGA, builds a dictionary containing n-grams of legitimate domain names and trains a wasserstein generative adversarial network (WGAN) to synthesize domain names by merging the sampled n-grams. In our implementation, we set the embedding size of n-grams to 5,000. The domain synthesizer and the discriminator are implemented using the residual convolutional network.

# Disclaimer
This project is still under development and may be missing at the moment. In addition, some paths may require you to modify.
