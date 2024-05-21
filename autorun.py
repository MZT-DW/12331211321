'''
Author: your name
Date: 2024-05-21 20:26:24
LastEditTime: 2024-05-21 21:19:22
LastEditors: your name
Description: 
FilePath: \GESR\autorun.py
可以输入预定的版权声明、个性签名、空行等
'''

from PyGP import Dataset
from PyGP import dataset_dict as d, dataset_save
from glob import glob
import os

import argparse
import numpy as np

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Autorun a method.", add_help=False)
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-run_time', dest='run', type=int, default=5,
                        help='the run time')
    parser.add_argument('-job_id', dest='job_id', type=int, default=None,
                        help='the index of dataset')
    parser.add_argument('-save_file', dest='save_file', type=str, default='./result/PMLB_res_fri_test_nocrg.txt',
                        help='the index of dataset')
    parser.add_argument('-dataset_dir', dest='save_file', type=str, default='./Dataset/pmlb/datasets',
                        help='the index of dataset')
    args = parser.parse_args()


    dataset_dir='./Dataset/pmlb/datasets'
    if args.dataset_dir.endswith('.tsv.gz'):
        datasets = [args.dataset_dir]
    elif args.dataset_dir.endswith('*'):
        print('capturing glob', args.dataset_dir + '/*.tsv.gz')
        datasets = sorted(glob(args.dataset_dir + '*/*.tsv.gz'))
    else:
        datasets = sorted(glob(args.dataset_dir + '/*/*.tsv.gz'))
    dataset_size = len(datasets)

    for j in range(dataset_size):
        res_0 = []
        res_1 = []
        res_train = []
        R2_0 = []
        R2_1 = []
        exp_list = []
        dcurve_list = []
        keep_times = []
        next = False
        test_min = 100000000

        with open(args.save_file, 'a+') as f:
            f.write("\n{0}, {1}\n\n".format(j, datasets[j].split('\\')[-1].split('.tsv.gz')[0]))
        for i in range(args.run):
            command = 'python {SCRIPT}.py -run_id {RUNID} -job_id {JOBID} -file_name {FILE}'.format(SCRIPT='try', RUNID=i % args.run,JOBID=j, FILE=save_file)
            os.system(command)
