import os
import sys 
import numpy as np
import re
import pyBigWig
import argparse

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']

num_bp_grch37=[249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560]
chr_len_grch37={}
for i in np.arange(len(chr_all)):
    chr_len_grch37[chr_all[i]]=num_bp_grch37[i]

num_bp_grch38=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
chr_len_grch38={}
for i in np.arange(len(chr_all)):
    chr_len_grch38[chr_all[i]]=num_bp_grch38[i]

# argv
def get_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-i', '--input', default='GM449.bigwig', nargs='+', type=str, help='input bigwig (replicates)')
    parser.add_argument('-o', '--output', default='GM449.npy', type=str, help='output file name')
    parser.add_argument('-rg', '--ref_genome', default='grch37', type=str, help='reference genome')
    args = parser.parse_args()
    return args

args=get_args()
print(args)

if args.ref_genome=='grch37':
    chr_len = chr_len_grch37
    num_bp = num_bp_grch37
else: # grch37 & grch38 only for now
    chr_len = chr_len_grch38
    num_bp = num_bp_grch38

# chromosome-specific subsampling seeds
chr_to_seed={}
i=0
for the_chr in chr_all:
    chr_to_seed[the_chr]=i
    i+=1

# temparory sample files for quantile normalization
path0='./output/'
os.system('mkdir -p ' + path0)
file_output = path0 + args.output

list_input=args.input

# subsampling; multiple replicates are added; step_size is the subsampling ratio
step_size=1000
sample_len=np.ceil(np.array(num_bp)/step_size).astype(int)
sample=np.zeros(sum(sample_len))
start=0
j=0
for the_chr in chr_all:
    print(the_chr)
    for the_input in list_input:
        bw = pyBigWig.open(the_input)
        signal=np.array(bw.values(the_chr,0,chr_len[the_chr]))
        signal[np.isnan(signal)]=0 # set nan to 0
        # random index
        np.random.seed(chr_to_seed[the_chr])
        index=np.random.randint(0,chr_len[the_chr],sample_len[j])
        # subsample
        sample[start:(start+sample_len[j])] += signal[index]
        bw.close()
    start+=sample_len[j]
    j+=1

if np.any(np.isnan(sample)):
    print('wtf! sample contains nan!')
sample.sort()
np.save(file_output, sample)


