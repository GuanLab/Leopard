import os
import sys
import numpy as np
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
    parser.add_argument('-i', '--input', default='K562.bigwig', nargs='+', type=str, help='input DNase-seq bigwig')
    parser.add_argument('-o', '--output', default='avg.bigwig', type=str, help='output file name')
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

input_all=args.input

bw_output = pyBigWig.open(args.output,'w')
bw_output.addHeader(list(zip(chr_all , num_bp)), maxZooms=0) # zip two turples

for the_chr in chr_all:
    print('calculating average for ' + the_chr)
    x = np.zeros(chr_len[the_chr])
    for the_input in input_all:
        print('loading ' + the_input)
        bw=pyBigWig.open(the_input)
        x += bw.values(the_chr, 0, chr_len[the_chr])
        bw.close()
    x=x/len(input_all)
    ## convert to bigwig format
    # pad two zeroes
    z=np.concatenate(([0],x,[0]))
    # find boundary
    starts=np.where(np.diff(z)!=0)[0]
    ends=starts[1:]
    starts=starts[:-1]
    vals=x[starts]
    if starts[0]!=0:
        ends=np.concatenate(([starts[0]],ends))
        starts=np.concatenate(([0],starts))
        vals=np.concatenate(([0],vals))
    if ends[-1]!=chr_len[the_chr]:
        starts=np.concatenate((starts,[ends[-1]]))
        ends=np.concatenate((ends,[chr_len[the_chr]]))
        vals=np.concatenate((vals,[0]))
    # write 
    chroms = np.array([the_chr] * len(vals))
    bw_output.addEntries(chroms, starts, ends=ends, values=vals)

bw_output.close()


