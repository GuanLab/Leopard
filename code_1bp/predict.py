#!/home/hyangl/anaconda3/bin/python
import pyBigWig
import argparse
import os
import sys
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) ## disable tf warning!
stderr = sys.stderr # disable printing "Using TensorFlow backend."
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
from keras import backend as K
import unet
from util.auc import score_record, calculate_auc

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

###### PARAMETER ###############

size=2**11*5 # 10240
num_channel=6
write_pred=True # whether generate .vec prediction file
size_edge=int(100) # chunk edges to be excluded
batch=100

## prepare label and image ############################################3
chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=np.array([249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560])

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

path_computer='../data/'
path1=path_computer + 'dna_bigwig/' # dna
path2=path_computer + 'dnase_bigwig/' # dnase

# argv
def get_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-m', '--model', default='weights_1.h5', type=str,
        help='model name')
    parser.add_argument('-tf', '--transcription_factor', default='CTCF', type=str,
        help='transcript factor')
    parser.add_argument('-te', '--test', default='K562', type=str,
        help='test cell type')
    parser.add_argument('-chr', '--chromosome', default='chr21', nargs='+', type=str,
        help='test chromosome')
    parser.add_argument('-para', '--parallel', default=1, type=int,
        help='control GPU memory usage when running multiple models in parallel') 
    args = parser.parse_args()
    return args

args=get_args()

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1/float(args.parallel) - 0.05
set_session(tf.Session(config=config))

#print(sys.argv)
name_model=args.model
the_tf=args.transcription_factor
cell_test=args.test
list_chr=args.chromosome

model = unet.get_unet(the_lr=1e-3,num_class=1,num_channel=num_channel,size=size)
model.load_weights(name_model)
#model.summary()

# open bigwig
list_dna=['A','C','G','T']
dict_dna={}
for the_id in list_dna:
    dict_dna[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
feature_avg=pyBigWig.open(path2 + 'avg.bigwig')
feature_test=pyBigWig.open(path2 + cell_test + '.bigwig')
############

if __name__ == '__main__':

    the_name=name_model.split('/')[-1].split('.')[0]

    for the_chr in list_chr:
        print(the_chr)
        output_all=np.zeros(chr_len[the_chr])
        count_all=np.zeros(chr_len[the_chr])

        for phase in np.arange(0,1,0.5):
            i = 0 + int(size * phase)
            if (i!=0): # shift the end as well
                d1 = chr_len[the_chr] - size + int(size * phase)
            else:
                d1 = chr_len[the_chr]

            while (i<d1):
                start = i
                end = i + size*batch
                if (end>d1):
                    end = d1
                    start = d1 - size*batch
                image = np.zeros((num_channel, size*batch))
                # dna
                num=0
                for j in np.arange(len(list_dna)):
                    the_id=list_dna[j]
                    image[num,:] = dict_dna[the_id].values(the_chr,start,end)
                    num+=1
                # feature & diff
                image[num,:] = np.array(feature_test.values(the_chr, start, end))
                avg = np.array(feature_avg.values(the_chr, start, end))
                image[num+1,:] = image[num,:] - avg

                ## make predictions ################
                input_pred=np.reshape(image.T,(batch,size,num_channel))
                output1 = model.predict(input_pred)
                output1 = np.reshape(output1,(size*batch, 1)).T
                output_new=output1.flatten()

                i_batch=0
                while (i_batch<batch):
                    i_start = start + i_batch*size
                    i_end = i_start + size
                    if (i_start==0):
                        start_new = i_start
                        end_new = i_end - size_edge
                        start_tmp = 0 + i_batch*size
                        end_tmp = size - size_edge + i_batch*size
                    elif (i_end==d1):
                        start_new = i_start + size_edge
                        end_new = i_end
                        start_tmp = size_edge + i_batch*size
                        end_tmp = size + i_batch*size
                    else:
                        start_new = i_start + size_edge
                        end_new = i_end - size_edge
                        start_tmp = size_edge + i_batch*size
                        end_tmp = size - size_edge + i_batch*size
                    output_all[start_new:end_new] += output_new[start_tmp:end_tmp]
                    count_all[start_new:end_new] += 1
                    i_batch += 1

                i=i+int(size*batch)

        del output1
        del output_new
        del image
        del input_pred

        ######################################################################
        
        # scoring
        output_all=np.divide(output_all,count_all)

        if write_pred:
            os.system('mkdir -p output')
            np.save('./output/pred_' + the_tf + '_' + cell_test + '_' + the_chr + '_' + the_name, output_all)


for the_id in list_dna:
    dict_dna[the_id].close()
feature_avg.close()
feature_test.close()













