import copy
import numpy as np
from nemo.utils import logging
from collections import deque


class Token:
    def __init__(self, state, dist=0.0, start_frame=None):
        self.state = state
        self.dist = dist     
        self.alive = True
        self.start_frame = start_frame

class WBHyp:
    def __init__(self, word, score, start_frame, end_frame, tokenization):
        self.word = word
        self.score = score
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.tokenization = tokenization
        

def beam_pruning(next_tokens, threshold):
    if not next_tokens:
        return []   
    alive_tokens = [token for token in next_tokens if token.alive]
    best_token = alive_tokens[np.argmax([token.dist for token in alive_tokens])]
    next_tokens = [token for token in alive_tokens if token.dist > best_token.dist - threshold]
    return next_tokens


def state_pruning(next_tokens):
    if not next_tokens:
        return []
    # hyps pruning
    for token in next_tokens:
        if not token.state.best_token:
            token.state.best_token = token
        else:
            if token.dist <= token.state.best_token.dist:
                token.alive = False
            else:
                token.state.best_token.alive = False
                token.state.best_token = token
    next_tokens = [token for token in next_tokens if token.alive]

    # clean best_tokens in context_graph
    for token in next_tokens:
        token.state.best_token = None
    
    return next_tokens


def find_best_hyp(spotted_words):

    clusters_dict = {}
    for hyp in spotted_words:
        hl, hr = hyp.start_frame, hyp.end_frame
        h_cluster_name = f"{hl}_{hr}"
        insert_cluster = True

        for cluster in clusters_dict:
            cl, cr = int(cluster.split("_")[0]), int(cluster.split("_")[1])
            # in case of intersection:
            if cl <= hl <= cr or cl <= hr <= cr or hl <= cl <= hr or hl <= cr <= hr:
                if hyp.score > clusters_dict[cluster].score:
                    clusters_dict.pop(cluster)
                    insert_cluster = True
                    break
                else:
                    insert_cluster = False         
        if insert_cluster:
            clusters_dict[h_cluster_name] = hyp
    
    best_hyp_list = [clusters_dict[cluster] for cluster in clusters_dict]        
    
    return best_hyp_list


def get_ctc_word_alignment(logprob, model, token_weight=1.0):
    
    alignment_ctc = np.argmax(logprob, axis=1)
    # logging.warning("---------------------")
    # logging.warning(f"alignment_ctc is: {alignment_ctc}")

    # get token alignment
    token_alignment = []
    prev_idx = None
    for i, idx in enumerate(alignment_ctc):
        if idx != model.decoder.blank_idx:
            token = model.tokenizer.ids_to_tokens([int(idx)])[0]
            if idx == prev_idx:
                token_alignment.pop()
            token_alignment.append((token, i, logprob[i, idx].item()))
        prev_idx = idx
    
    # get word alignment
    slash = "▁"
    word_alignment = []
    word = ""
    l, r, score = None, None, None
    token_boost = token_weight
    for item in token_alignment:
        if not word:
            word = item[0][1:]
            l = item[1]
            r = item[1]
            score = item[2]+token_boost
        else:
            if item[0].startswith(slash):
                word_alignment.append((word, l, r, score))
                word = item[0][1:]
                l = item[1]
                r = item[1]
                score = item[2]+token_boost
            else:
                word += item[0]
                r = item[1]
                score += item[2]+token_boost
    word_alignment.append((word, l, r, score))
    
    if len(word_alignment) == 1 and not word_alignment[0][0]:
        word_alignment = []
    
    return word_alignment


def filter_wb_hyps(best_hyp_list, word_alignment):
    
    # logging.warning("---------------------")
    # logging.warning(f"word_alignment is: {word_alignment}")
    if not word_alignment:
        return best_hyp_list

    best_hyp_list_new = []
    current_frame = 0
    for hyp in best_hyp_list:
        lh, rh = hyp.start_frame, hyp.end_frame
        for i in range(current_frame, len(word_alignment)):
            item = word_alignment[i]
            li, ri = item[1], item[2]
            if li <= lh <= ri or li <= rh <= ri or lh <= li <= rh or lh <= ri <= rh:
                if hyp.score >= item[3]:
                    best_hyp_list_new.append(hyp)
                current_frame = i
                break
    
    return best_hyp_list_new


# def filter_wb_hyps(best_hyp_list, word_alignment):
    
#     best_hyp_list_new = []
#     current_spot = 0
#     for hyp in best_hyp_list:
#         lh, rh = hyp.start_frame, hyp.end_frame
#         overall_spot_score = 0
#         spotted = False
#         for i in range(current_spot, len(word_alignment)):
#             item = word_alignment[i]
#             li, ri = item[1], item[2]
#             if li <= lh <= ri or li <= rh <= ri or lh <= li <= rh or lh <= ri <= rh:
#                 overall_spot_score += item[3]
#                 spotted = True
#             elif spotted and hyp.score >= overall_spot_score:
#                 best_hyp_list_new.append(hyp)
#                 current_spot = i-1
#                 break
    
#     return best_hyp_list_new


def recognize_wb(logprobs, context_graph, asr_model, beam_threshold=None, context_score=0.0, keyword_thr=-3, ctc_ali_token_weight=2.0):
    start_state = context_graph.root
    active_tokens = []
    next_tokens = []
    spotted_words = []

    blank_thr = np.log(0.80)

    for frame in range(logprobs.shape[0]):
        active_tokens.append(Token(start_state))
        logprob_frame = logprobs[frame]
        best_dist = None
        for token in active_tokens:
            ## skip blank for first token if root:
            if token.state is context_graph.root and logprobs[frame][asr_model.decoder.blank_idx] > blank_thr:
                continue
            ## end skip blank
            for transition_state in token.state.next:

                ### running beam (start):
                if transition_state != asr_model.decoder.blank_idx:
                    current_dist = token.dist + logprob_frame[int(transition_state)].item() + context_score
                else:
                    current_dist = token.dist + logprob_frame[int(transition_state)].item()
                
                if not best_dist:
                    best_dist = current_dist
                else:
                    if current_dist < best_dist - beam_threshold:
                        continue
                    elif current_dist > best_dist:
                        best_dist = current_dist
                ### running beam (end)

                new_token = Token(token.state.next[transition_state], current_dist, token.start_frame)

                if not new_token.start_frame:
                    new_token.start_frame = frame

                # if end of word:
                if new_token.state.is_end and new_token.dist > keyword_thr:
                    word = asr_model.tokenizer.ids_to_text(new_token.state.word)
                    spotted_words.append(WBHyp(word, new_token.dist, new_token.start_frame, frame, new_token.state.word))
                    if len(new_token.state.next) == 1:
                        if current_dist is best_dist:
                            best_dist = None
                        continue
                next_tokens.append(new_token)
                # else:
                #     next_tokens.append(new_token)
        # state and beam prunings:
        next_tokens = beam_pruning(next_tokens, beam_threshold)
        next_tokens = state_pruning(next_tokens)
        # print(f"frame step is: {frame}")
        # print(f"number of active_tokens is: {len(next_tokens)}")

        active_tokens = next_tokens
        next_tokens = []

    # find best hyp for spotted keywords:
    best_hyp_list = find_best_hyp(spotted_words)
    print(f"---spotted words:")
    for hyp in best_hyp_list:
        print(f"{hyp.word}: [{hyp.start_frame};{hyp.end_frame}], score:{hyp.score:-.2f}")
    
    # filter wb hyps according to greedy ctc predictions
    ctc_word_alignment = get_ctc_word_alignment(logprobs, asr_model, token_weight=ctc_ali_token_weight)
    best_hyp_list_new = filter_wb_hyps(best_hyp_list, ctc_word_alignment)
    print("---final result is:")
    for hyp in best_hyp_list_new:
        print(f"{hyp.word}: [{hyp.start_frame};{hyp.end_frame}], score:{hyp.score:-.2f}")


    return best_hyp_list_new