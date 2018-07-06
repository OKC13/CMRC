'''
Evaluation script for CMRC 2018
Version 1.0
'''
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import nltk


# split Chinese with English
def mixed_segmentation(in_str):
	in_str = in_str.lower().strip()
	segs_out = []
	temp_str = ""
	for char in in_str:
		if re.search(ur'[\u4e00-\u9fa5]', char) or char in ['-',':','_','*','^','/','\\','~','`','+','=']:
			if temp_str != "":
				ss = nltk.word_tokenize(temp_str)
				segs_out.extend(ss)
				temp_str = ""
			segs_out.append(char)
		else:
			temp_str += char
	return segs_out 


# find longest common string
def find_lcs(s1, s2):
	m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
	mmax = 0
	p = 0
	for i in range(len(s1)):
		for j in range(len(s2)):
			if s1[i] == s2[j]:
				m[i+1][j+1] = m[i][j]+1
				if m[i+1][j+1] > mmax:
					mmax=m[i+1][j+1]
					p=i+1
	return s1[p-mmax:p], mmax

#
def evaluate(ground_truth_file, prediction_file):
	f1 = 0
	em = 0
	total_count = 0
	for instance in ground_truth_file:
		context_id   = instance['context_id'].strip()
		context_text = instance['context_text'].strip()
		for qas in instance['qas']:
			total_count += 1
			query_id    = qas['query_id'].strip()
			query_text  = qas['query_text'].strip()
			answers 	= qas['answers']

			if query_id not in prediction_file:
				sys.stderr.write('Unanswered question: {}\n'.format(query_id))
				continue

			prediction 	= prediction_file[query_id]
			f1 += calc_f1_score(answers, prediction)
			em += calc_em_score(answers, prediction)

	f1_score = 100.0 * f1 / total_count
	em_score = 100.0 * em / total_count
	return f1_score, em_score

#
def calc_f1_score(answers, prediction):
	f1_scores = []
	for ans in answers:
		ans = ans.strip()
		ans_segs = mixed_segmentation(ans)
		prediction_segs = mixed_segmentation(prediction)
		lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
		if lcs_len == 0:
			f1_scores.append(0)
			continue
		precision 	= 1.0*lcs_len/len(prediction_segs)
		recall 		= 1.0*lcs_len/len(ans_segs)
		f1 			= (2*precision*recall)/(precision+recall)
		f1_scores.append(f1)
	return max(f1_scores)

#
def calc_em_score(answers, prediction):
	em = 0
	for ans in answers:
		ans_ = ans.lower().strip()
		prediction_ = prediction.lower().strip()
		if ans_ == prediction_:
			em = 1
			break
	return em

#
if __name__ == '__main__':
	ground_truth_file   = json.load(open(sys.argv[1], 'rb'))
	prediction_file     = json.load(open(sys.argv[2], 'rb'))
	F1, EM = evaluate(ground_truth_file, prediction_file)
	AVG = (EM+F1)*0.5
	print('AVG: {:.3f} F1: {:.3f} EM: {:.3f} FILE: {}'.format(AVG, F1, EM, sys.argv[2]))
