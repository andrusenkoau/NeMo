import copy
import numpy as np

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
    alive_tokens = [token for token in next_tokens if token.alive]
    best_token = alive_tokens[np.argmax([token.dist for token in alive_tokens])]
    next_tokens = [token for token in alive_tokens if token.dist > best_token.dist - threshold]
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


def recognize_wb(logprobs, context_graph, asr_model, beam_threshold=None, context_score=0.0, keyword_thr=-3):
    start_state = context_graph.root
    active_tokens = []
    next_tokens = []
    spotted_words = []

    for frame in range(logprobs.shape[0]):
        active_tokens.append(Token(start_state))
        logprob_frame = logprobs[frame]
        for token in active_tokens:
            for transition_state in token.state.next:
                # parent_dist = copy.deepcopy(token.dist)
                # new_token = Token(token.state.next[transition_state], parent_dist, token.start_frame)
                new_token = Token(token.state.next[transition_state], token.dist, token.start_frame)
                new_token.dist += logprob_frame[int(transition_state)].item()
                # boost hyp only if token is not blank
                if transition_state != asr_model.decoder.blank_idx:
                    new_token.dist += context_score
                # define start_frame
                if not new_token.start_frame:
                    new_token.start_frame = frame

                # if end of word:
                if new_token.state.is_end and new_token.dist > keyword_thr:
                    word = asr_model.tokenizer.ids_to_text(new_token.state.word)
                    spotted_words.append(WBHyp(word, new_token.dist, new_token.start_frame, frame, new_token.state.word))
                else:
                    next_tokens.append(new_token)
        # state and beam prunings:
        # next_tokens = state_pruning(next_tokens)
        next_tokens = beam_pruning(next_tokens, beam_threshold)
        # print(f"frame step is: {frame}")
        # print(f"number of active_tokens is: {len(next_tokens)}")

        active_tokens = next_tokens
        next_tokens = []

    # find best hyp for spotted keywords:
    best_hyp_list = find_best_hyp(spotted_words)
    print(f"---spotted words:")
    for hyp in best_hyp_list:
        print(f"{hyp.word}: [{hyp.start_frame};{hyp.end_frame}], score:{hyp.score:-.2f}")
            

    return best_hyp_list