import sys
import os

import cPickle as pickle

from cfr.logger import Logger as Log
Log.VERBOSE = True

import cfr.evaluation as evaluation
from cfr.plotting import *

def sort_by_config(results, configs, key):
    vals = np.array([cfg[key] for cfg in configs])
    I_vals = np.argsort(vals)

    for k in results['train'].keys():
        results['train'][k] = results['train'][k][I_vals,]
        results['valid'][k] = results['valid'][k][I_vals,]

        if k in results['test']:
            results['test'][k] = results['test'][k][I_vals,]

    configs_sorted = []
    for i in I_vals:
        configs_sorted.append(configs[i])

    return results, configs_sorted

def evaluate(path, dataset, overwrite=False, filters=None):
    if not os.path.isdir(path):
        raise Exception('Could not find output at path: %s' % path)

    output_dir = path

    if dataset==1:
        data_train = '../data/LaLonde/jobs_dw_bin.train.npz'
        data_test = '../data/LaLonde/jobs_dw_bin.test.npz'
        binary = True
    elif dataset==2:
        data_train = '../data/LaLonde/jobs_dw_bin.new.10.train.npz'
        data_test = '../data/LaLonde/jobs_dw_bin.new.10.test.npz'
        binary = True
    elif dataset==3:
        data_train = '../data/LaLonde/jobs_DW_bin.bias.married.10.train.npz'
        data_test = '../data/LaLonde/jobs_DW_bin.bias.married.10.test.npz'
        binary = True
    elif dataset==4:
        data_train = '../data/LaLonde/jobs_DW_bin.bias.nodegr.10.train.npz'
        data_test = '../data/LaLonde/jobs_DW_bin.bias.nodegr.10.test.npz'
        binary = True
    elif dataset==5:
        data_train = '../data/ihdp/ihdp_imb/ihdp_imb_p0_400_1_1-500.npz'
        data_test = '../data/ihdp/ihdp_imb/ihdp_imb_p0_400_1_1-500.npz'
        binary = False
    elif dataset==6:
        data_train = '../data/ihdp/ihdp_imb/ihdp_imb_p0_400_2_1-500.npz'
        data_test = '../data/ihdp/ihdp_imb/ihdp_imb_p0_400_2_1-500.npz'
        binary = False
    elif dataset==7:
        data_train = '../data/ihdp/ihdp_imb/ihdp_imb_p0_400_3_1-500.npz'
        data_test = '../data/ihdp/ihdp_imb/ihdp_imb_p0_400_3_1-500.npz'
        binary = False
    elif dataset==8:
        data_train = '../data/LaLonde/jobs_DW_bin.bias.nodegr.25.train.npz'
        data_test = '../data/LaLonde/jobs_DW_bin.bias.nodegr.25.test.npz'
        binary = True
    else:
        data_train = '../data/ihdp/ihdp_npci_1-1000.train.npz'
        data_test = '../data/ihdp/ihdp_npci_1-1000.test.npz'
        binary = False


    # Evaluate results
    eval_path = '%s/evaluation.npz' % output_dir
    if overwrite or (not os.path.isfile(eval_path)):
        eval_results, configs = evaluation.evaluate(output_dir,
                                data_path_train=data_train,
                                data_path_test=data_test,
                                binary=binary)
        # Save evaluation
        pickle.dump((eval_results, configs), open(eval_path, "wb"))
    else:
        if Log.VERBOSE:
            print 'Loading evaluation results from %s...' % eval_path
        # Load evaluation
        eval_results, configs = pickle.load(open(eval_path, "rb"))

    # Sort by alpha
    #eval_results, configs = sort_by_config(eval_results, configs, 'p_alpha')

    # Print evaluation results
    if binary:
        plot_evaluation_bin(eval_results, configs, output_dir, data_train, data_test, filters)
    else:
        plot_evaluation_cont(eval_results, configs, output_dir, data_train, data_test, filters)

    # Plot evaluation
    #if configs[0]['loss'] == 'log':
    #    plot_cfr_evaluation_bin(eval_results, configs, output_dir)
    #else:
    #    plot_cfr_evaluation_cont(eval_results, configs, output_dir)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print 'Usage: python evaluate.py <output folder> <dataset> <overwrite> <filters (optional)>'
    else:
        dataset = int(sys.argv[2])

        overwrite = False
        if len(sys.argv)>3 and sys.argv[3] == '1':
            overwrite = True

        filters = None
        if len(sys.argv)>4:
            filters = eval(sys.argv[4])

        evaluate(sys.argv[1], dataset, overwrite, filters=filters)
