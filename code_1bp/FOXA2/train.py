import pyBigWig
import argparse
import os
import sys
import numpy as np
import re
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import keras
import scipy.io
print('tf-' + tf.__version__, 'keras-' + keras.__version__)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45 
set_session(tf.Session(config=config))
import random
from datetime import datetime
import unet

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=np.array([249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560])

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

size=2**11*5 # 10240
num_channel=6

num_sample=100000
batch_size=100

path_computer='../../data/'
path1=path_computer + 'dna_bigwig/' # dna
path2=path_computer + 'dnase_bigwig/' # dnase
path3=path_computer + 'chipseq_conservative_refine_bigwig/' # label

# argv
def get_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-tf', '--transcription_factor', default='CTCF', type=str,
        help='transcript factor')
    parser.add_argument('-tr', '--train', default='K562', type=str,
        help='train cell type')
    parser.add_argument('-vali', '--validate', default='A549', type=str,
        help='validate cell type')
    parser.add_argument('-par', '--partition', default='1', type=str,
        help='chromasome parition')
    args = parser.parse_args()
    return args

args=get_args()

print(sys.argv)
the_tf=args.transcription_factor
cell_train=args.train
cell_vali=args.validate
par=args.partition 

## random seed for chr partition
chr_train_all=['chr2','chr3','chr4','chr5','chr6','chr7','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr22','chrX']
ratio=0.5
np.random.seed(int(par))
np.random.shuffle(chr_train_all)
tmp=int(len(chr_train_all)*ratio)
chr_set1=chr_train_all[:tmp]
chr_set2=chr_train_all[tmp:]
print(chr_set1)
print(chr_set2)

name_model='weights_' + cell_train + '_' + cell_vali + '_' + par + '.h5'
model = unet.get_unet(the_lr=1e-3,num_class=1,num_channel=num_channel,size=size)
#model.load_weights(name_model)
model.summary()

### random seed; use train cell for now
#cell_all=['A549','GM12878','H1-hESC','HCT116','HeLa-S3','HepG2','IMR-90','induced_pluripotent_stem_cell','K562','liver','MCF-7','Panc1','PC-3']
#cell_to_seed={}
#i=0
#for the_cell in cell_all:
#    cell_to_seed[the_cell]=i
#    i+=1
#if cell_train in cell_to_seed.keys():
#    np.random.seed(cell_to_seed[cell_train])
#else:
#    pass

## sample index for chunks ###########
tmp=[]
for the_chr in chr_set1:
    tmp.append(chr_len[the_chr])
freq=np.rint(np.array(tmp)/sum(tmp)*1000).astype('int')
index_set1=np.array([])
for i in np.arange(len(chr_set1)):
    index_set1 = np.hstack((index_set1, np.array([chr_set1[i]] * freq[i])))
np.random.shuffle(index_set1)
tmp=[]
for the_chr in chr_set2:
    tmp.append(chr_len[the_chr])
freq=np.rint(np.array(tmp)/sum(tmp)*1000).astype('int')
index_set2=np.array([])
for i in np.arange(len(chr_set2)):
    index_set2 = np.hstack((index_set2, np.array([chr_set2[i]] * freq[i])))
np.random.shuffle(index_set2)
#############################################

# open bigwig
list_dna=['A','C','G','T']
dict_dna={}
for the_id in list_dna:
    dict_dna[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
feature_avg=pyBigWig.open(path2 + 'avg.bigwig')
feature_train=pyBigWig.open(path2 + cell_train + '.bigwig')
feature_vali=pyBigWig.open(path2 + cell_vali + '.bigwig')
label_train=pyBigWig.open(path3 + the_tf + '_' + cell_train + '.bigwig')
label_vali=pyBigWig.open(path3 + the_tf + '_' + cell_vali + '.bigwig')
############

##### augmentation parameters ######
if_time=False
max_scale=1.15
min_scale=1
if_mag=True
#if_mag=False
max_mag=1.15
min_mag=0.9
if_flip=False
####################################

def generate_data(batch_size, if_train):
    i=0
    j=0
    while True:
        b = 0
        image_batch = []
        label_batch = []

        while b < batch_size:
            if (if_train==1):
                if i == len(index_set1):
                    i=0
                    np.random.shuffle(index_set1)
                the_chr=index_set1[i]
                feature_bw = feature_train
                label_bw = label_train
                i+=1
            else:
                if j == len(index_set2):
                    j=0
                    np.random.shuffle(index_set2)
                the_chr=index_set2[j]
                feature_bw = feature_vali
                label_bw = label_vali
                j+=1

            start=int(np.random.randint(0, chr_len[the_chr] - size, 1))
            end=start + size

            label = np.array(label_bw.values(the_chr, start, end))

            image = np.zeros((num_channel, size))
            num=0
            for k in np.arange(len(list_dna)):
                the_id=list_dna[k]
                image[num,:] = dict_dna[the_id].values(the_chr, start, end)
                num+=1
            image[num,:] = np.array(feature_bw.values(the_chr, start, end))
            avg = np.array(feature_avg.values(the_chr, start, end))
            image[num+1,:] = image[num,:] - avg

            # augmentation
            if (if_train==1) and (if_mag):
                rrr=random.random()
                rrr_mag=rrr*(max_mag-min_mag)+min_mag
                image[4,:]=image[4,:]*rrr_mag                               

            image_batch.append(image.T)
            label_batch.append(label.T)

            b += 1

        image_batch=np.array(image_batch)
        label_batch=np.array(label_batch)
        yield image_batch, label_batch


callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join('./', name_model),
    save_weights_only=False,monitor='val_loss')
    ]

model.fit_generator(
    generate_data(batch_size,True),
    steps_per_epoch=int(num_sample // batch_size), nb_epoch=5,
    validation_data=generate_data(batch_size,False),
    validation_steps=int(num_sample // batch_size),callbacks=callbacks,verbose=1)

for the_id in list_dna:
    dict_dna[the_id].close()
feature_avg.close()
feature_train.close()
feature_vali.close()
label_train.close()
label_vali.close()

