# cfrnet
Counterfactual Regression using Balancing Neural Networks as developed by Johansson, Shalit & Sontag (2016) and Shalit, Johansson & Sontag (2016). cfrnet is implemented in Python using TensorFlow and NumPy.

# Code

The core components of cfrnet, i.e. the TensorFlow graph, is contained in cfr_net.py.
A simple training script is contained in cfr_train_simple.py. This file takes many flag parameters, all of which are listed at the top of the file. 

# Examples

In the root directory there is an example script called run_simple.sh which calls the python script cfr_train_simple.py. This example trains a counterfactual regression model on a single realization of the simulated IHDP data (see references), contained in data/ihdp_sample.csv. It creates a folder called results/single_\<config & timestamp\>/ which contains 5 files: 

* config.txt - The configuration used for the run
* log.txt - A log file
* loss.csv - The objective, factual, counterfactual and imbalance losses over time
* y_pred.csv - The predicted factual and counterfactual outputs for all units)
* results.npz - A numpy array file with the fields "pred" and "loss" which contains the same output as the previous two files.

The data (.csv) file has the following columns: treatment (0 or 1), y_factual, y_cfactual, mu0, mu1, x_1, …, x_d. 
mu0 and mu1 are not used in training (they are the true simulated outcomes under control and treatment, without noise). 

# References
Uri Shalit, Fredrik D. Johansson & David Sontag. [Bounding and Minimizing Counterfactual Error](https://arxiv.org/abs/1606.03976), arXiv:1606.03976 Preprint, 2016

Fredrik D. Johansson, Uri Shalit &  David Sontag. [Learning Representations for Counterfactual Inference](http://jmlr.org/proceedings/papers/v48/johansson16.pdf). 33rd International Conference on Machine Learning (ICML), June 2016.


