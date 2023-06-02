# Introduction

This repository contains the implementations of PDKGA and its competitors. PKDGA is an adversarial domain generation algorithm that uses partial knowledge of the target detector for reinforcement learning training. Its overall achitecture is depected as follows. 
<p align="center">
 <img src="https://github.com/abcdefdf/PKDGA/assets/98793069/c289c3ee-d8d1-4061-83dd-0958e1c84fda" />
<p>
 
# Installation and Usage Hints
PKDGA is created, trained and evaluated on Ubuntu 16.04 or Win10 using Python.

# DGA Reproducibility
## PKDGA
### Create a anaconda environment:

*conda create -n <environment_name> pytohn==3.7*  //**environment_name** is the name of the created environment 

### Install the required packages:

*pip -r install requirement.txt*

### Pre-train the target domain detector
 Select a target detector and train it to detect malicious domains. The detailed command for training each detector will be shown in the **Detector Reproducibilition** part.

The screenshots during pretraining target detector：
<p align="center">
 <img src="https://github.com/abcdefdf/PKDGA/assets/98793069/6c389df8-cc9c-4b81-b755-4f007cd2863b" />
<p>
  
### Training PKDGA to compromise the target detector
python pkdga/gen_model.py

The screenshots during training PKDGA is

<p align="center">
 <img src="https://github.com/abcdefdf/PKDGA/assets/98793069/f00f3f3f-f3ed-4314-a7bd-72969678a1ba" />
<p> 

### Generate domains using PKDGA

*ython gen_sample.py*

Some samples(2rd-level domains only) are shown as follows:
  
<streflix\>
<notgichtirnax\>
<ecawerdaille\>
<porete\>
<cridmi\>
<ifireda\>
<xlazairan\>
<youregan\>
<pathrovegalima\>
<porcedch\>
<omehmswime\>
<pardesleatress\>
<migoo\>
<fintstalnotic\>
<zpus-animesops\>
<hatporhsaviaker\>
<yudija\>
<dinyrimesbot\>
<elinkbilyimen1\>
<latorb\>
<jahas\>
<sbagebissear\>
<bashiammy\>
<loguo\>
<notebonlop\>
<intarasbing\>
<otoulaa\>
<byirbaotive\>
<tvhjemborodop\>
<tenihahang\>\>
<nrfwedreeke\>
<mnex\>
<mreelcagazun\>
<pred\>
<latana\>
<clive-tipanews\>
<lowpek\>
<voithh\>
<petcolf\>
<svvertenexvedes\>
<gastrafro\>

## The usages of the rest files in *PKDGA/* are introduced as follows. 

### rollout.py: 
  
The fuction of computing policy-based gradient, which makes the token sequence Y resist against target detector D. It is called by the *gen_model.py* when training PKDGA.

### exp1.py:
 The experimental code of measuring DGAs' evasoin ability (the results in TABLE IV)

### exp2.py: 
 
The experimental code of measuring training sets' impacts on PKDGA (the results in Fig.9).  

 
## khaos

### Introduction: 

DGA, builds a dictionary containing n-grams of legitimate domain names and trains a wasserstein generative adversarial network (WGAN) to synthesize domain names by merging the sampled n-grams. In our implementation, we set the embedding size of n-grams to 5,000. The domain synthesizer and the discriminator are implemented using the residual convolutional network.

### Usage: 
 
 *python khaos/gan_language.py*

# Detector Reproducibility
## RF
 
### Introduction: 

 DGA detector, it extracts 21 lightweight DGA features, including structural, linguistic and statistical features, and employs a supervised model (Random Forest) for classification.
###Usage: 

 *Python RF/RF_classifier.py*

## bilstm,lstm,textcnn
### Introduction: 
 DGA detector, directly leverage neural networks to distinguish AGDs from benign ones without extracting features manually.

### Usage:
 
*python lstm/lstm_class.py -layer -emb -hid -device* // **layer** is the layer number of the network, **emb** is the embedding size, **hid** is the size of hidden layer.
 
*python textcnn/cnn.py*  -device
 
*python bilstm/bilstm.py -max_features - maxlen -device*  // **maxlen** is the maximal length of domain names.
 
## graph
### Introduction: 
 
Graph is able to detect dictionary-based AGDs that are generated by concatenating words from a predefined dictionary. It constructs a graph describing the relationships between the largest common substrings in a domain and the graph with average degrees larger than a threshold is considered as malicious. In our implementation, the substrings that repeat more than three times are considered as basic words, and the degree threshold is set to 3.

### Usage:
 
 *python graph/WordGraph.py*
 
## Statics
### Introduction:  
DGA detector, argues that AGDs follow different distributions from the legitimate domain names. A domain is identified as benign if it has a negligible distance from the legitimate domains. In particular, it measures the distance using Kullback-Leibler divergence, Jaccard index and Edit distance.

### Usage:
 *python statics/statics.py*
 
# Disclaimer
This project is still under development and may be missing at the moment. In addition, some paths may require you to modify.
