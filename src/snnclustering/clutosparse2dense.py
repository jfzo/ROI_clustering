#!/usr/bin/python

'''
Converts dense CSV file into a sparse matrix file in Cluto's format.

Input file must be a comma separated file that also contains labels in the last field.
'''

import sys
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o","--outfile", type=str, help="Path to the output file where the sparse matrix will be stored")
parser.add_argument("-i","--inputfile", type=str, help="Path to the input file in CSV format where the dense vectors are stored")
args = parser.parse_args()

if not args.inputfile or not args.outfile:
    print "Usage: ./clutosparse2dense -i <path to the input CSV file> -o <path to output file>"
    sys.exit()

outputfile = args.outfile
inputfile = args.inputfile


outfh = open(outputfile,'w')
infh = open(inputfile)

N,D,_ = map(int, infh.readline().strip().split(" "))
for L in infh:
    tuples  =L.strip().split(" ")
    row_v = np.zeros((1,D))
    for ix in range(0, len(tuples), 2):
        row_v[0, int(tuples[ix]) - 1] = float(tuples[ix+1])
    # write the vector to the output file
    outfh.write( "{:.5f}".format(row_v[0, 0]) )
    for i in range(1,D):
        outfh.write( ",{:.5f}".format(row_v[0, i]) )
    outfh.write("\n")


infh.close()
outfh.close()
