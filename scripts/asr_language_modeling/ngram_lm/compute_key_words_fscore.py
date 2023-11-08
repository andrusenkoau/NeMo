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
        if pair[1] in key_words:
            ref.append(pair[0].upper()) 
            hyp.append(pair[1].upper()) 
        else:
            ref.append(pair[0])
            hyp.append(pair[1])
    print(" ")
    print(f"ID: {os.path.basename(audio_filepath)}")
    print(f"REF: {' '.join(ref)}")
    print(f"HYP: {' '.join(hyp)}")
    

def compute_fscore(recognition_results_manifest, key_words_list, print_ali=True):

    data = load_data(recognition_results_manifest)
    key_words_set = set(key_words_list)
    
    key_words_dict = {}
    for item in key_words_list:
        item = item.split()
        if len(item) == 1:
            key_words_dict[item[0]] = None
        else:
            key_words_dict[item[0]] = item[1:]
    
    key_words_stat = {}
    for word in key_words_list:
        key_words_stat[word] = [0, 0, 0] # [tp, totall, fp]

    gt, fp, tp = 0, 0, 0
    eps = '***'

    for item in data:
        audio_filepath = item['audio_filepath']
        ref = item['text'].split()
        hyp = item['pred_text'].split()
        ali = align(ref, hyp, eps)
        recognized_words = []
        false_positive_words = []
        for idx, pair in enumerate(ali):
            phrase_match = True
            if pair[0] in key_words_dict:
                if not key_words_dict[pair[0]]:
                    gt += 1
                    key_words_stat[pair[0]][1] += 1
                    if pair[0] == pair[1]:
                        tp += 1
                        recognized_words.append(pair[0])
                        key_words_stat[pair[0]][0] += 1
                else:
                    phrase_ref = pair[0]
                    phrase_hyp = pair[1]
                    i = idx
                    for next_word in key_words_dict[pair[0]]:
                        if (i+1) >= len(ali) or next_word != ali[i+1][0]:
                            phrase_match = False
                            break
                        phrase_ref += f" {next_word}"
                        phrase_hyp += f" {ali[i+1][1]}"
                        i += 1
                    if phrase_match:
                        gt += 1
                        key_words_stat[phrase_ref][1] += 1
                        if phrase_ref == phrase_hyp:
                            tp += 1
                            recognized_words.append(phrase_ref)
                            key_words_stat[phrase_ref][0] += 1
                    
                    
            # processing of false positive recognition: 
            if pair[1] in key_words_dict and pair[0] != pair[1]:
                if not key_words_dict[pair[1]]:
                    fp += 1
                    key_words_stat[pair[1]][2] += 1
                    false_positive_words.append(pair[1])
                else:
                    i = idx
                    phrase_hyp = pair[1]
                    for next_word in key_words_dict[pair[1]]:
                        if (i+1) >= len(ali) or next_word != ali[i+1][1]:
                            phrase_match = False
                            break
                        phrase_hyp += f" {next_word}"
                        i += 1
                    if phrase_match:
                        fp += 1
                        key_words_stat[phrase_hyp][2] += 1
                        false_positive_words.append(phrase_hyp)
                    
        if recognized_words and print_ali:
            # print_alignment(audio_filepath, ali, recognized_words)
            print_alignment(audio_filepath, ali, false_positive_words)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (gt + 1e-8)
    fscore = 2*(precision*recall)/(precision+recall + 1e-8)

    print("\n"+"***"*15)
    print("Per words statistic (word: correct/totall (false positive)):\n")
    max_len = max([len(x) for x in key_words_stat if key_words_stat[x][1] > 0 or key_words_stat[x][2] > 0])
    for word in key_words_list:
        if key_words_stat[word][1] > 0 or key_words_stat[word][2] > 0:
            false_positive = ""
            if key_words_stat[word][2] > 0:
                false_positive = key_words_stat[word][2]
            print(f"{word:>{max_len}}: {key_words_stat[word][0]:3}/{key_words_stat[word][1]:<3} {false_positive:>3}")
    print("***"*15)

    print(" ")
    print("***"*10)
    print(f"Precision: {precision:.4f} ({tp}/{tp + fp}) fp:{fp}")
    print(f"Recall:    {recall:.4f} ({tp}/{gt})")
    print(f"Fscore:    {fscore:.4f}")
    print("***"*10) 



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_manifest", type=str, required=True, help="manifest with recognition results",
    )
    parser.add_argument(
        "--key_words_file", type=str, required=True, help="file of key words for fscore calculation"
    )

    args = parser.parse_args()
    #key_words_list = [x for x in args.key_words_list.split('_')]
    key_words_list = []
    for line in open(args.key_words_file).readlines():
        item = line.strip().split("-")[1].lower()
        if item not in key_words_list:
            key_words_list.append(item)
    compute_fscore(args.input_manifest, key_words_list)


if __name__ == '__main__':
    main()
