import tensorflow as tf
import numpy as np
import sys, os
import getopt
import random
import datetime
import cfr_net as cfr
import traceback

''' Define parameter flags '''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('n_in', 2, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 2, """Number of regression layers. """)
tf.app.flags.DEFINE_float('p_alpha', 1e-4, """Imbalance regularization param. """)
tf.app.flags.DEFINE_float('p_lambda', 0.0, """Regularization parameter. """)
tf.app.flags.DEFINE_float('dropout_in', 1.0, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 1.0, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_float('lrate', 0.05, """Learning rate. """)
tf.app.flags.DEFINE_float('decay', 0.5, """RMSProp decay. """)
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
tf.app.flags.DEFINE_integer('dim_in', 100, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
tf.app.flags.DEFINE_integer('experiments', 100, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 2000, """Number of iterations. """)
tf.app.flags.DEFINE_float('weight_init', 0.01, """Weight initialization scale. """)
tf.app.flags.DEFINE_string('outdir', '../results/tfnet_topic/alpha_sweep_22_d100/', """Output directory. """)
tf.app.flags.DEFINE_string('datapath', '../data/topic/csv/', """Data directory. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.95, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.app.flags.DEFINE_integer('use_p_correction', 1, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_string('imb_fun', 'mmd_lin', """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
tf.app.flags.DEFINE_integer('output_csv',1, """Whether to save a CSV file with the results""")
tf.app.flags.DEFINE_integer('output_delay', 200, """Number of iterations between outputs. """)
tf.app.flags.DEFINE_integer('wass_iterations', 20, """Number of iterations in Wasserstein computation. """)
tf.app.flags.DEFINE_float('wass_lambda', 1, """Wasserstein lambda. """)
tf.app.flags.DEFINE_integer('wass_bpt', 0, """Backprop through T matrix? """)
tf.app.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)

NUM_ITERATIONS_PER_DECAY = 100

def train(outdir):
    HAVE_TRUTH = False

    ''' Set up paths and start log '''
    npzfile = outdir+'result'
    repfile = outdir+'reps'
    outform = outdir+'y_pred'
    lossform = outdir+'loss'
    logfile = outdir+'log.txt'
    f = open(logfile,'w')
    f.close()

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir+'config.txt')
    log(logfile, 'Training with hyperparameters: alpha=%.2e, lambda=%.2e' % (FLAGS.p_alpha,FLAGS.p_lambda))

    ''' Load data '''
    log(logfile, 'Loading data for dimensions... '+FLAGS.datapath)
    x_all, t_all, y_f_all, y_cf_all = load_data(FLAGS.datapath)
    if not y_cf_all is None:
        HAVE_TRUTH = True
    dim = x_all.shape[1]
    n = x_all.shape[0]

    log(logfile, 'Loaded data with shape [%d,%d]' % (n,dim))

    ''' Start Session '''
    log(logfile, 'Starting session...')
    sess = tf.Session()

    ''' Initialize input placeholders '''
    x_  = tf.placeholder("float", shape=[None,dim], name='x_') # Features
    t_  = tf.placeholder("float", shape=[None,1], name='t_')   # Treatent
    y_ = tf.placeholder("float", shape=[None,1], name='y_')  # Outcome

    ''' Parameter placeholders '''
    alpha_ = tf.placeholder("float", name='alpha_')
    lambda_ = tf.placeholder("float", name='lambda_')
    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    p = tf.placeholder("float", name='p_treated')

    ''' Define model graph '''
    log(logfile, 'Defining graph...')
    dims = [dim,FLAGS.dim_in,FLAGS.dim_out]
    CFR = cfr.cfr_net(x_, t_, y_, p, FLAGS, alpha_, lambda_, do_in, do_out, dims)

    if FLAGS.varsel:
        w_proj = tf.placeholder("float", shape=[dim], name='w_proj')
        projection = CFR.weights_in[0].assign(w_proj)

    ''' Set up optimizer '''
    log(logfile, 'Training...')
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(FLAGS.lrate, global_step, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)
    train_step = tf.train.RMSPropOptimizer(lr, FLAGS.decay).minimize(CFR.tot_loss,global_step=global_step)

    ''' Compute treatment probability'''
    t_cf_all = 1-t_all
    if FLAGS.use_p_correction:
        p_treated = np.mean(t_all)
    else:
        p_treated = 0.5

    ''' Set up loss feed_dicts'''
    dict_factual = {x_: x_all, t_: t_all, y_: y_f_all, \
        do_in:1.0, do_out:1.0, alpha_:FLAGS.p_alpha, \
        lambda_:FLAGS.p_lambda, p:p_treated}

    if HAVE_TRUTH:
        dict_cfactual = {x_: x_all, t_: t_cf_all, y_: y_cf_all, \
            do_in:1.0, do_out:1.0}

    ''' Initialize tensorflow variables '''
    sess.run(tf.initialize_all_variables())

    ''' Compute losses before training'''
    losses = []
    obj_loss, f_error, imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, \
        CFR.imb_loss], feed_dict=dict_factual)

    cf_error = np.nan
    if HAVE_TRUTH:
        cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

    losses.append([obj_loss, f_error, cf_error, imb_err])

    log(logfile, 'Objective Factual CFactual Imbalance')
    log(logfile, str(losses[0]))

    ''' Train for m iterations '''
    for i in range(FLAGS.iterations):

        ''' Fetch sample '''
        I = random.sample(range(0, n), FLAGS.batch_size)
        x_batch = x_all[I,:]
        t_batch = t_all[I]
        y_batch = y_f_all[I]

        ''' Do one step of gradient descent '''
        sess.run(train_step, feed_dict={x_: x_batch, t_: t_batch, \
            y_: y_batch, do_in:FLAGS.dropout_in, do_out:FLAGS.dropout_out, \
            alpha_:FLAGS.p_alpha, lambda_:FLAGS.p_lambda, p:p_treated})

        ''' Project variable selection weights '''
        if FLAGS.varsel:
            wip = cfr.simplex_project(sess.run(CFR.weights_in[0]), 1)
            sess.run(projection,feed_dict={w_proj: wip})

        ''' Compute loss every N iterations '''
        if i % FLAGS.output_delay == 0:
            obj_loss,f_error,imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_loss],
                feed_dict=dict_factual)

            y_pred = sess.run(CFR.output, feed_dict={x_: x_batch, t_: t_batch, \
                y_: y_batch, do_in:FLAGS.dropout_in, do_out:FLAGS.dropout_out, \
                alpha_:FLAGS.p_alpha, lambda_:FLAGS.p_lambda, p:p_treated})

            cf_error = np.nan
            if HAVE_TRUTH:
                cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

            losses.append([obj_loss, f_error, cf_error, imb_err])
            loss_str = str(i) + '\tObj: %.4g,\tF: %.4g,\tCf: %.4g,\tImb: %.4g' % (obj_loss, f_error, cf_error, imb_err)

            if FLAGS.loss == 'log':
                y_pred = 1.0*(y_pred>0.5)
                acc = 100*(1-np.mean(np.abs(y_batch-y_pred)))
                loss_str += ',\tAcc: %.2f%%' % acc

            log(logfile, loss_str)

    log(logfile, 'Ending learning rate: %.2g' % sess.run(lr))

    ''' Predict response and store '''
    ypred_f = sess.run(CFR.output, feed_dict={x_: x_all, t_: t_all, \
        do_in:1.0, do_out:1.0, alpha_:FLAGS.p_alpha, lambda_:FLAGS.p_lambda})
    ypred_c = sess.run(CFR.output, feed_dict={x_: x_all, t_: t_cf_all, \
        do_in:1.0, do_out:1.0, alpha_:FLAGS.p_alpha, lambda_:FLAGS.p_lambda})

    ypred = np.concatenate((ypred_f,ypred_c),axis=1)

    log(logfile, 'Saving to %s...' % outform)
    if FLAGS.output_csv:
        np.savetxt('%s.csv' % (outform), ypred, delimiter=',')
        np.savetxt('%s.csv' % (lossform), losses, delimiter=',')

    ''' Compute weights'''
    if FLAGS.varsel:
        all_weights = np.dstack((all_weights,sess.run(CFR.weights_in[0])))
        all_beta = np.dstack((all_beta,sess.run(CFR.weights_pred)))

    ''' Save results and predictions '''
    if FLAGS.varsel:
        np.savez(npzfile, pred=ypred, loss=losses, w=all_weights, beta=all_beta)
    else:
        np.savez(npzfile, pred=ypred, loss=losses)

    ''' Save representations '''
    if FLAGS.save_rep:
        reps = sess.run([CFR.h_rep], feed_dict={x_: x_all, do_in:1.0, do_out:0.0})
        np.savez(repfile, rep=reps )

def log(logfile,str):
    """ Log and print string """
    with open(logfile,'a') as f:
        f.write(str+'\n')
    print str

def save_config(fname):
    """ Save configuration """
    flagdict =  FLAGS.__dict__['__flags']
    s = '\n'.join(['%s: %s' % (k,str(flagdict[k])) for k in sorted(flagdict.keys())])
    f = open(fname,'w')
    f.write(s)
    f.close()

def load_data(fname):
    """ Load data set """
    data = np.loadtxt(open(fname,"rb"),delimiter=",")
    x = data[:,5:]

    t = data[:,0:1]
    yf = data[:,1:2]
    ycf = data[:,2:3]

    return x,t,yf,ycf

def main(argv=None):
    """ Runs the main training method """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir+'/'

    try:
        train(outdir)
    except Exception as e:
        with open(outdir+'error.txt','w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise

if __name__ == '__main__':
    tf.app.run()
