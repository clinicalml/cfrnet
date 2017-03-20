import numpy as np
import os

from logger import Logger as Log
from loader import *

POL_CURVE_RES = 40

class NaNException(Exception):
    pass


def policy_range(n, res=10):
    step = int(float(n)/float(res))
    n_range = range(0,int(n+1),step)
    if not n_range[-1] == n:
        n_range.append(n)

    # To make sure every curve is same length. Incurs a small error if res high.
    # Only occurs if number of units considered differs.
    # For example if resampling validation sets (with different number of
    # units in the randomized sub-population)

    while len(n_range) > res:
        k = np.random.randint(len(n_range)-2)+1
        del n_range[k]

    return n_range

def policy_val(t, yf, eff_pred, compute_policy_curve=False):
    """ Computes the value of the policy defined by predicted effect """

    if np.any(np.isnan(eff_pred)):
        return np.nan, np.nan

    policy = eff_pred>0
    treat_overlap = (policy==t)*(t>0)
    control_overlap = (policy==t)*(t<1)

    if np.sum(treat_overlap)==0:
        treat_value = 0
    else:
        treat_value = np.mean(yf[treat_overlap])

    if np.sum(control_overlap)==0:
        control_value = 0
    else:
        control_value = np.mean(yf[control_overlap])

    pit = np.mean(policy)
    policy_value = pit*treat_value + (1-pit)*control_value

    policy_curve = []

    if compute_policy_curve:
        n = t.shape[0]
        I_sort = np.argsort(-eff_pred)

        n_range = policy_range(n, POL_CURVE_RES)

        for i in n_range:
            I = I_sort[0:i]

            policy_i = 0*policy
            policy_i[I] = 1
            pit_i = np.mean(policy_i)

            treat_overlap = (policy_i>0)*(t>0)
            control_overlap = (policy_i<1)*(t<1)

            if np.sum(treat_overlap)==0:
                treat_value = 0
            else:
                treat_value = np.mean(yf[treat_overlap])

            if np.sum(control_overlap)==0:
                control_value = 0
            else:
                control_value = np.mean(yf[control_overlap])

            policy_curve.append(pit_i*treat_value + (1-pit_i)*control_value)

    return policy_value, policy_curve

def pdist2(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*X.dot(Y.T)
    nx = np.sum(np.square(X),1,keepdims=True)
    ny = np.sum(np.square(Y),1,keepdims=True)
    D = (C + ny.T) + nx

    return np.sqrt(D + 1e-8)

def cf_nn(x, t):
    It = np.array(np.where(t==1))[0,:]
    Ic = np.array(np.where(t==0))[0,:]

    x_c = x[Ic,:]
    x_t = x[It,:]

    D = pdist2(x_c, x_t)

    nn_t = Ic[np.argmin(D,0)]
    nn_c = It[np.argmin(D,1)]

    return nn_t, nn_c

def pehe_nn(yf_p, ycf_p, y, x, t, nn_t=None, nn_c=None):
    if nn_t is None or nn_c is None:
        nn_t, nn_c = cf_nn(x,t)

    It = np.array(np.where(t==1))[0,:]
    Ic = np.array(np.where(t==0))[0,:]

    ycf_t = 1.0*y[nn_t]
    eff_nn_t = ycf_t - 1.0*y[It]
    eff_pred_t = ycf_p[It] - yf_p[It]

    eff_pred = eff_pred_t
    eff_nn = eff_nn_t

    '''
    ycf_c = 1.0*y[nn_c]
    eff_nn_c = ycf_c - 1.0*y[Ic]
    eff_pred_c = ycf_p[Ic] - yf_p[Ic]

    eff_pred = np.vstack((eff_pred_t, eff_pred_c))
    eff_nn = np.vstack((eff_nn_t, eff_nn_c))
    '''

    pehe_nn = np.sqrt(np.mean(np.square(eff_pred - eff_nn)))

    return pehe_nn

def evaluate_bin_att(predictions, data, i_exp, I_subset=None,
                     compute_policy_curve=False, nn_t=None, nn_c=None):

    x = data['x'][:,:,i_exp]
    t = data['t'][:,i_exp]
    e = data['e'][:,i_exp]
    yf = data['yf'][:,i_exp]
    yf_p = predictions[:,0]
    ycf_p = predictions[:,1]

    att = np.mean(yf[t>0]) - np.mean(yf[(1-t+e)>1])

    if not I_subset is None:
        x = x[I_subset,:]
        t = t[I_subset]
        e = e[I_subset]
        yf_p = yf_p[I_subset]
        ycf_p = ycf_p[I_subset]
        yf = yf[I_subset]

    yf_p_b = 1.0*(yf_p>0.5)
    ycf_p_b = 1.0*(ycf_p>0.5)

    if np.any(np.isnan(yf_p)) or np.any(np.isnan(ycf_p)):
        raise NaNException('NaN encountered')

    #IMPORTANT: NOT USING BINARIZATION FOR EFFECT, ONLY FOR CLASSIFICATION!

    eff_pred = ycf_p - yf_p;
    eff_pred[t>0] = -eff_pred[t>0];

    ate_pred = np.mean(eff_pred[e>0])
    atc_pred = np.mean(eff_pred[(1-t+e)>1])

    att_pred = np.mean(eff_pred[(t+e)>1])
    bias_att = att_pred - att

    err_fact = np.mean(np.abs(yf_p_b-yf))

    p1t = np.mean(yf[t>0])
    p1t_p = np.mean(yf_p[t>0])

    lpr = np.log(p1t / p1t_p + 0.001)

    policy_value, policy_curve = \
        policy_val(t[e>0], yf[e>0], eff_pred[e>0], compute_policy_curve)

    pehe_appr = pehe_nn(yf_p, ycf_p, yf, x, t, nn_t, nn_c)

    return {'ate_pred': ate_pred, 'att_pred': att_pred,
            'bias_att': bias_att, 'atc_pred': atc_pred,
            'err_fact': err_fact, 'lpr': lpr,
            'policy_value': policy_value, 'policy_risk': 1-policy_value,
            'policy_curve': policy_curve, 'pehe_nn': pehe_appr}

def evaluate_cont_ate(predictions, data, i_exp, I_subset=None,
    compute_policy_curve=False, nn_t=None, nn_c=None):

    x = data['x'][:,:,i_exp]
    t = data['t'][:,i_exp]
    yf = data['yf'][:,i_exp]
    ycf = data['ycf'][:,i_exp]
    mu0 = data['mu0'][:,i_exp]
    mu1 = data['mu1'][:,i_exp]
    yf_p = predictions[:,0]
    ycf_p = predictions[:,1]

    if not I_subset is None:
        x = x[I_subset,]
        t = t[I_subset]
        yf_p = yf_p[I_subset]
        ycf_p = ycf_p[I_subset]
        yf = yf[I_subset]
        ycf = ycf[I_subset]
        mu0 = mu0[I_subset]
        mu1 = mu1[I_subset]

    eff = mu1-mu0

    rmse_fact = np.sqrt(np.mean(np.square(yf_p-yf)))
    rmse_cfact = np.sqrt(np.mean(np.square(ycf_p-ycf)))

    eff_pred = ycf_p - yf_p;
    eff_pred[t>0] = -eff_pred[t>0];

    ite_pred = ycf_p - yf
    ite_pred[t>0] = -ite_pred[t>0]
    rmse_ite = np.sqrt(np.mean(np.square(ite_pred-eff)))

    ate_pred = np.mean(eff_pred)
    bias_ate = ate_pred-np.mean(eff)

    att_pred = np.mean(eff_pred[t>0])
    bias_att = att_pred - np.mean(eff[t>0])

    atc_pred = np.mean(eff_pred[t<1])
    bias_atc = atc_pred - np.mean(eff[t<1])

    pehe = np.sqrt(np.mean(np.square(eff_pred-eff)))

    pehe_appr = pehe_nn(yf_p, ycf_p, yf, x, t, nn_t, nn_c)

    # @TODO: Not clear what this is for continuous data
    #policy_value, policy_curve = policy_val(t, yf, eff_pred, compute_policy_curve)

    return {'ate_pred': ate_pred, 'att_pred': att_pred,
            'atc_pred': atc_pred, 'bias_ate': bias_ate,
            'bias_att': bias_att, 'bias_atc': bias_atc,
            'rmse_fact': rmse_fact, 'rmse_cfact': rmse_cfact,
            'pehe': pehe, 'rmse_ite': rmse_ite, 'pehe_nn': pehe_appr}
            #'policy_value': policy_value, 'policy_curve': policy_curve}

def evaluate_result(result, data, validation=False,
        multiple_exps=False, binary=False):

    predictions = result['pred']

    if validation:
        I_valid = result['val']

    n_units, _, n_rep, n_outputs = predictions.shape

    #@TODO: Should depend on parameter
    compute_policy_curve = True

    eval_results = []
    #Loop over output_times
    for i_out in range(n_outputs):
        eval_results_out = []

        if not multiple_exps and not validation:
            nn_t, nn_c = cf_nn(data['x'][:,:,0], data['t'][:,0])


        #Loop over repeated experiments
        for i_rep in range(n_rep):

            if validation:
                I_valid_rep = I_valid[i_rep,:]
            else:
                I_valid_rep = None

            if multiple_exps:
                i_exp = i_rep
                if validation:
                    nn_t, nn_c = cf_nn(data['x'][I_valid_rep,:,i_exp], data['t'][I_valid_rep,i_exp])
                else:
                    nn_t, nn_c = cf_nn(data['x'][:,:,i_exp], data['t'][:,i_exp])
            else:
                i_exp = 0

            if validation and not multiple_exps:
                nn_t, nn_c = cf_nn(data['x'][I_valid_rep,:,i_exp], data['t'][I_valid_rep,i_exp])

            if binary:
                eval_result = evaluate_bin_att(predictions[:,:,i_rep,i_out],
                    data, i_exp, I_valid_rep, compute_policy_curve, nn_t=nn_t, nn_c=nn_c)
            else:
                eval_result = evaluate_cont_ate(predictions[:,:,i_rep,i_out],
                    data, i_exp, I_valid_rep, compute_policy_curve, nn_t=nn_t, nn_c=nn_c)

            eval_results_out.append(eval_result)

        eval_results.append(eval_results_out)

    # Reformat into dict
    eval_dict = {}
    keys = eval_results[0][0].keys()
    for k in keys:
        arr = [[eval_results[i][j][k] for i in range(n_outputs)] for j in range(n_rep)]
        v = np.array([[eval_results[i][j][k] for i in range(n_outputs)] for j in range(n_rep)])
        eval_dict[k] = v

    # Gather loss
    # Shape [times, types, reps]
    # Types: obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj
    if 'loss' in result.keys() and result['loss'].shape[1]>=6:
        losses = result['loss']
        n_loss_outputs = losses.shape[0]

        if validation:
            objective = np.array([losses[(n_loss_outputs*i)/n_outputs,6,:] for i in range(n_outputs)]).T
        else:
            objective = np.array([losses[(n_loss_outputs*i)/n_outputs,0,:] for i in range(n_outputs)]).T

        eval_dict['objective'] = objective

    return eval_dict

def evaluate(output_dir, data_path_train, data_path_test=None, binary=False):

    print '\nEvaluating experiment %s...' % output_dir

    # Load results for all configurations
    results = load_results(output_dir)

    if len(results) == 0:
        raise Exception('No finished results found.')

    # Separate configuration files
    configs = [r['config'] for r in results]

    # Test whether multiple experiments (different data)
    multiple_exps = (configs[0]['experiments'] > 1)
    if Log.VERBOSE and multiple_exps:
        print 'Multiple data (experiments) detected'

    # Load training data
    if Log.VERBOSE:
        print 'Loading TRAINING data %s...' % data_path_train
    data_train = load_data(data_path_train)

    # Load test data
    if data_path_test is not None:
        if Log.VERBOSE:
            print 'Loading TEST data %s...' % data_path_test
        data_test = load_data(data_path_test)
    else:
        data_test = None



    # Evaluate all results
    eval_results = []
    configs_out = []
    i = 0
    if Log.VERBOSE:
        print 'Evaluating result (out of %d): ' % len(results)
    for result in results:
        if Log.VERBOSE:
            print 'Evaluating %d...' % (i+1)

        try:
            eval_train = evaluate_result(result['train'], data_train,
                validation=False, multiple_exps=multiple_exps, binary=binary)

            eval_valid = evaluate_result(result['train'], data_train,
                validation=True, multiple_exps=multiple_exps, binary=binary)

            if data_test is not None:
                eval_test = evaluate_result(result['test'], data_test,
                    validation=False, multiple_exps=multiple_exps, binary=binary)
            else:
                eval_test = None

            eval_results.append({'train': eval_train, 'valid': eval_valid, 'test': eval_test})
            configs_out.append(configs[i])
        except NaNException as e:
            print 'WARNING: Encountered NaN exception. Skipping.'
            print e

        i += 1

    # Reformat into dict
    eval_dict = {'train': {}, 'test': {}, 'valid': {}}
    keys = eval_results[0]['train'].keys()
    for k in keys:
        v = np.array([eval_results[i]['train'][k] for i in range(len(eval_results))])
        eval_dict['train'][k] = v

        v = np.array([eval_results[i]['valid'][k] for i in range(len(eval_results))])
        eval_dict['valid'][k] = v

        if eval_test is not None and k in eval_results[0]['test']:
            v = np.array([eval_results[i]['test'][k] for i in range(len(eval_results))])
            eval_dict['test'][k] = v

    return eval_dict, configs_out
