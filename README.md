# cfrnet
Counterfactual regression (CFR) by learning balanced representations, as developed by Johansson, Shalit & Sontag (2016) and Shalit, Johansson & Sontag (2016). cfrnet is implemented in Python using TensorFlow and NumPy.

# Code

The core components of cfrnet, i.e. the TensorFlow graph, is contained in cfr/cfr_net.py. A simple training script is contained example_ihdp.sh. This file runs the model on the IHDP data with parameters supplied by configs/example_ihdp.txt.

# Examples

- TO BE WRITTEN --

# References
Uri Shalit, Fredrik D. Johansson & David Sontag. [Estimating individual treatment effect: generalization bounds and algorithms] (https://arxiv.org/abs/1606.03976), arXiv:1606.03976 Preprint, 2016

Fredrik D. Johansson, Uri Shalit &  David Sontag. [Learning Representations for Counterfactual Inference](http://jmlr.org/proceedings/papers/v48/johansson16.pdf). 33rd International Conference on Machine Learning (ICML), June 2016.
