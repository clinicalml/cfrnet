# cfrnet
Counterfactual regression (CFR) by learning balanced representations, as developed by Johansson, Shalit & Sontag (2016) and Shalit, Johansson & Sontag (2016). cfrnet is implemented in Python using TensorFlow 0.12.0-rc1 and NumPy 1.11.3. The code has _not_ been tested with TensorFlow 1.0.

# Code

The core components of cfrnet, i.e. the TensorFlow graph, is contained in cfr/cfr_net.py. The training is performed by cfr_net_train.py. The file cfr_param_search.py takes a configuration file as input and allows the user to randomly sample from the supplied parameters (given that there are multiple values given in a list. See configs/example_ihdp.txt for an example.

A typical experiment uses cfr_param_search.py and evaluate.py as sub-routines. cfr_param_search is best used to randomly search the parameter space. In the output directory, it creates a log of which configurations have been used so far, so that the same experiment is not repeated. evaluate.py goes through the predictions produced by the model and evaluates the error.

## cfr_param_search
Usage: 

# Examples

A simple experiment example is contained in example_ihdp.sh. This file runs the model on (a subset of) the IHDP data with parameters supplied by configs/example_ihdp.txt.

# References
Uri Shalit, Fredrik D. Johansson & David Sontag. [Estimating individual treatment effect: generalization bounds and algorithms] (https://arxiv.org/abs/1606.03976), arXiv:1606.03976 Preprint, 2016

Fredrik D. Johansson, Uri Shalit &  David Sontag. [Learning Representations for Counterfactual Inference](http://jmlr.org/proceedings/papers/v48/johansson16.pdf). 33rd International Conference on Machine Learning (ICML), June 2016.
