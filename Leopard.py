import os
import numpy as np
import glob
import argparse
from subprocess import Popen

def get_args():
    parser = argparse.ArgumentParser(description="run Leopard predictions")
    parser.add_argument('-tf', '--transcription_factor', default='HNF4A', type=str,
        help='transcript factor')
    parser.add_argument('-te', '--test', default='liver', type=str,
        help='test cell type')
    parser.add_argument('-chr', '--chromosome', default='chr21', nargs='+', type=str,
        help='test chromosome')
    parser.add_argument('-m', '--mode', default='fast', type=str,
        help='prediction mode (complete or fast)')
    parser.add_argument('-reso', '--resolution', default='1bp', type=str,
        help='resolution of prediction')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    the_tf = args.transcription_factor
    the_test = args.test
    chr_all = args.chromosome
#    print(chr_all)
    if not isinstance(args.chromosome, list):
        chr_all = [chr_all] # if one chr, convert it into list
#    print(chr_all)
    the_reso = args.resolution

    # list existing models
    the_path = 'code_' + the_reso + '/' + the_tf + '/'
    models = glob.glob(the_path + 'weights_*1.h5')
    _,the_train,the_vali,_ = models[0].split('/')[-1].split('_')
    num_par = len(glob.glob(the_path + 'weights_' + the_train + '_' + the_vali + '_*.h5'))

    if args.mode!='complete':
        num_par = 1
        print('fast mode - run ' + str(num_par) + ' replicate')
    else:
        print('complete mode - run ' + str(num_par) + ' replicates trained on different seeds')

    # run prediction in parallel
    for i in np.arange(1,num_par+1):
        print('run replicate ' + str(i) + '/' + str(num_par) + ':')
        cmd_all=[]
        for j in np.arange(len(models)):
            _,the_train,the_vali,_ = models[j].split('/')[-1].split('_')
            the_model = the_path + 'weights_' + the_train + '_' + the_vali + '_' + str(i) + '.h5'
            print('model: ' + the_train + '_' + the_vali)
            cmd = ['python', 'code_' + the_reso + '/predict.py', '-m', the_model, \
                '-tf',  the_tf, '-te', the_test, '-chr'] + chr_all + \
                ['-para', str(len(models))]
            cmd_all.append(cmd)
        procs = [ Popen(i) for i in cmd_all ]
        for p in procs: # run in parallel
            p.wait()

    # stacking prediction from multiple models
    print('combining predictions from different models')
    for the_chr in chr_all:
        print(the_chr)
        the_name = './output/pred_' + the_tf + '_' + the_test + '_' + the_chr
        for i in np.arange(1,num_par + 1):
            for j in np.arange(len(models)):
                _,the_train,the_vali,_ = models[j].split('/')[-1].split('_')
                if i==1 and j==0:
                    pred = np.load(the_name + '_weights_' + the_train + '_' + the_vali + '_' + str(i) + '.npy')
                else:
                    pred += np.load(the_name + '_weights_' + the_train + '_' + the_vali + '_' + str(i) + '.npy')
        pred = pred / float(len(models)) / float(num_par)
        np.save(the_name, pred)
        # remove individual predictions from each model
        for i in np.arange(1,num_par + 1):
            for j in np.arange(len(models)):
                _,the_train,the_vali,_ = models[j].split('/')[-1].split('_')
                os.system('rm ' + the_name + '_weights_' + the_train + '_' + the_vali + '_' + str(i) + '.npy')


if __name__ == '__main__':
    main()



