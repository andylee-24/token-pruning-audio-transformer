# -*- coding: utf-8 -*-
# @Time    : 11/15/20 1:04 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_esc_result.py

# summarize esc 5-fold cross validation result

import argparse
import numpy as np
import os
import glob

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_path", type=str, default='', help="the root path of the experiment")
parser.add_argument("--eval_file_pattern", type=str, default='best_result.csv', help="the file name of the result file")

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Recursively find all best_result.csv files under exp_path
    file_list = glob.glob(os.path.join(args.exp_path, '**', f'{args.eval_file_pattern}*'), recursive=True)

    
    if not file_list:
        print("No best_result.csv files found under", args.exp_path)
        exit(1)
    
    acc_sum = 0.0
    count = 0
    for file in file_list:
        result = np.loadtxt(file, delimiter=',')
        # Assuming the accuracy is in the second column (index 1)
        acc = result[1]
        acc_sum += acc
        # print('File {} accuracy: {:.4f}'.format(file, acc))
        count += 1

    avg_acc = acc_sum / count
    print('Average accuracy over {} files: {:.4f}'.format(count, avg_acc))
    
    # Write the average accuracy to a file
    output_file = os.path.join(args.exp_path, f'average_result_{avg_acc:.4f}-{args.eval_file_pattern}.txt')
    with open(output_file, 'w') as f:
        f.write('Average accuracy: {:.4f}'.format(avg_acc))