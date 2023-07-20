#!/usr/bin/env python

import argparse
import json
import os
from kaldialign import align


def load_data(manifest):
    data = []
    with open(manifest, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data


def print_alignment(audio_filepath, ali, key_words):
    ref, hyp = [], []
    for pair in ali:
        if pair[0] in key_words:
            ref.append(pair[0].upper()) 
            hyp.append(pair[1].upper()) 
        else:
            ref.append(pair[0])
            hyp.append(pair[1])
    print(" ")
    print(f"ID: {os.path.basename(audio_filepath)}")
    print(f"REF: {' '.join(ref)}")
    print(f"HYP: {' '.join(hyp)}")
    

def compute_fscore(recognition_results_manifest, key_words_list):

    data = load_data(recognition_results_manifest)
    key_words_set = set(key_words_list)
    key_words_stat = {}
    for word in key_words_set:
        key_words_stat[word] = [0, 0]

    gt, fn, fp, tn, tp = 0, 0, 0, 0, 0
    eps = '***'

    for item in data:
        audio_filepath = item['audio_filepath']
        ref = item['text'].split()
        hyp = item['pred_text'].split()
        ali = align(ref, hyp, eps)
        recognized_words = []
        for pair in ali:
            if pair[0] in key_words_set:
                gt += 1
                key_words_stat[pair[0]][-1] += 1
                if pair[0] == pair[1]:
                    tp += 1
                    recognized_words.append(pair[0])
                    key_words_stat[pair[0]][0] += 1
            if pair[1] in key_words_set:
                if pair[0] != pair[1]:
                    fp += 1
        if recognized_words:
            print_alignment(audio_filepath, ali, recognized_words)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (gt + 1e-8)
    fscore = 2*(precision*recall)/(precision+recall + 1e-8)

    print("\n"+"***"*15)
    print("Per words statistic (word: correct/totall):\n")
    max_len = max([len(x) for x in key_words_stat])
    for word in key_words_stat:
        print(f"{word:{max_len}}: {key_words_stat[word][0]}/{key_words_stat[word][-1]}")
    print("***"*15)

    print(" ")
    print("***"*10)
    print(f"Precision: {precision:.4f} ({tp}/{tp + fp})")
    print(f"Recall:    {recall:.4f} ({tp}/{gt})")
    print(f"Fscore:    {fscore:.4f}")
    print("***"*10) 



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_manifest", type=str, required=True, help="manifest with recognition results",
    )
    parser.add_argument(
        "--key_words_list", type=str, required=True, help="list of key words for fscore calculation"
    )

    args = parser.parse_args()
    key_words_list = [x for x in args.key_words_list.split(' ')]
    compute_fscore(args.input_manifest, key_words_list)


if __name__ == '__main__':
    main()
