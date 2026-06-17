# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from nemo.collections.asr.parts.context_biasing.context_graph_ctc import ContextGraphCTC, ContextState


@dataclass
class Token:
    """
    Dataclass of alignment tracking according to the Token Passing Algoritm (TPA).

    Args:
        state: state of Context-Biasing graph
        score: accumulated token score in log space
        start_frame: index of acoustic frame from which the token was created
        alive: token status (alive or dead)
    """

    state: ContextState
    score: float = 0.0
    start_frame: Optional[int] = None
    alive: bool = True


@dataclass
class WSHyp:
    """
    Hypothesis of Word Spotter prediction

    Args:
        word: spotted word
        score: accumulative score of best token
        start_frame: index of acoustic frame from which the best token was created
        end_frame: index of acoustic frame from which the final state of ContextGraph was reached
    """

    word: str
    score: float
    start_frame: int
    end_frame: int


def beam_pruning(next_tokens: List[Token], beam_threshold: float) -> List[Token]:
    """
    Prun all tokens whose score is worse than best_token.score - beam_threshold

    Args:
        next_tokens: list of input tokens
        beam_threshold: beam threshold

    Returns:
        list of pruned tokens
    """
    if not next_tokens:
        return []
    best_token = next_tokens[np.argmax([token.score for token in next_tokens])]
    next_tokens = [token for token in next_tokens if token.score > best_token.score - beam_threshold]
    return next_tokens


def state_pruning(next_tokens: List[Token]) -> List[Token]:
    """
    If there are several tokens on the same state, then leave only the best of them according to score

    Args:
        next_tokens: list of input tokens

    Returns:
        list of pruned tokens
    """
    if not next_tokens:
        return []
    # traverse all tokens and check each graph state for the best token
    for token in next_tokens:
        if not token.state.best_token:
            token.state.best_token = token
        else:
            if token.score <= token.state.best_token.score:
                token.alive = False
            else:
                token.state.best_token.alive = False
                token.state.best_token = token
    # save only alive tokens
    next_tokens_pruned = [token for token in next_tokens if token.alive]
    # clean all best_tokens in context_graph
    for token in next_tokens:
        token.state.best_token = None
    return next_tokens_pruned


def find_best_hyps(spotted_words: List[WSHyp], intersection_threshold: int = 10) -> List[WSHyp]:
    """
    Some spotted hypotheses may have overlap.
    If hypotheses intersection is greater than intersection_threshold,
    then the function leaves only the best hypothesis according to the score.

    Approach: sort by score (descending), greedily accept the best-scoring hyp first,
    then skip any subsequent hyp that overlaps with an already-accepted one.

    Args:
        spotted_words: list of spotter hypotheses WSHyp
        intersection_threshold: minimal intersection threshold (in percentages)

    Returns:
        list of best hyps without intersection, sorted by start_frame
    """
    if not spotted_words:
        return []

    sorted_hyps = sorted(spotted_words, key=lambda h: h.score, reverse=True)
    accepted = []
    accepted_intervals = []

    for hyp in sorted_hyps:
        hyp_interval = set(range(hyp.start_frame, hyp.end_frame + 1))
        overlaps = False
        for acc_interval in accepted_intervals:
            intersection_part = 100 / len(acc_interval) * len(hyp_interval & acc_interval)
            if intersection_part >= intersection_threshold:
                overlaps = True
                break
        if not overlaps:
            accepted.append(hyp)
            accepted_intervals.append(hyp_interval)

    accepted.sort(key=lambda h: h.start_frame)
    return accepted


def get_ctc_word_alignment(
    logprob: np.ndarray, asr_model, token_weight: float = 1.0, blank_idx: int = 0
) -> List[tuple]:
    """
    Get word level alignment (with start and end frames) based on argmax ctc predictions.
    The word score is a sum of non-blank token logprobs with additional token_weight.
    token_weight is used to prevent false accepts during filtering word spotting hypotheses.

    Args:
        logprob: ctc logprobs
        asr_model: asr model (ctc or hybrid transducer-ctc)
        token_weight: additional token weight for word-level ctc alignment

    Returns:
        list of word level alignment where each element is tuple (word, left_frame, rigth_frame, word_score)
    """

    alignment_ctc = np.argmax(logprob, axis=1)

    # get token level alignment
    token_alignment = []
    prev_idx = None
    for i, idx in enumerate(alignment_ctc):
        if idx != blank_idx:
            token = asr_model.tokenizer.ids_to_tokens([int(idx)])[0]
            if idx == prev_idx:
                prev_repeated_token = token_alignment.pop()
                token_logprob = prev_repeated_token[2] + logprob[i, idx].item()
                token_alignment.append((token, prev_repeated_token[1], token_logprob))
            else:
                token_alignment.append((token, i, logprob[i, idx].item()))
        prev_idx = idx

    # get word level alignment
    begin_of_word = "▁"
    word_alignment = []
    word = ""
    l, r, score = None, None, None
    for item in token_alignment:
        if not word:
            if item[0].startswith(begin_of_word):
                word = item[0][1:]
            else:
                word = item[0][:]
            l = item[1]
            r = item[1]
            score = item[2] + token_weight
        else:
            if item[0].startswith(begin_of_word):
                word_alignment.append((word, l, r, score))
                word = item[0][1:]
                l = item[1]
                r = item[1]
                score = item[2] + token_weight
            else:
                word += item[0]
                r = item[1]
                score += item[2] + token_weight
    if word:
        word_alignment.append((word, l, r, score))

    if len(word_alignment) == 1 and not word_alignment[0][0]:
        word_alignment = []

    return word_alignment


def filter_wb_hyps(best_hyp_list: List[WSHyp], word_alignment: List[tuple]) -> List[WSHyp]:
    """
    Compare scores of spotted words with overlapping words from ctc alignment.
    If score of spotted word is less than overalapping words from ctc alignment,
    the spotted word will removed as false positive.
    A spotted word may overlap with several words from ctc alignment ("gpu" -> "g p u").
    Here we use overall_spot_score variable to accumulate scores of several words.

    Args:
        best_hyp_list: list of spotted hypotheses WSHyp
        word_alignment: world level ctc alignment with word scores

    Returns:
        filtered best_hyp_list
    """

    if not word_alignment:
        return best_hyp_list

    best_hyp_list = sorted(best_hyp_list, key=lambda h: h.start_frame)
    best_hyp_list_filtered = []
    current_word_in_ali = 0
    for hyp in best_hyp_list:
        overall_spot_score = 0
        hyp_intersects = False
        hyp_interval = set(range(hyp.start_frame, hyp.end_frame + 1))
        # check if spotted word overlaps with words from ctc alignment
        for i in range(current_word_in_ali, len(word_alignment)):
            word_stats = word_alignment[i]
            word_interval = set(range(word_stats[1], word_stats[2] + 1))
            intersection_part = 100 / len(word_interval) * len(hyp_interval & word_interval)
            if intersection_part:
                if not hyp_intersects:
                    overall_spot_score = word_stats[3]
                else:
                    overall_spot_score += intersection_part / 100 * word_stats[3]
                hyp_intersects = True
            elif hyp_intersects:
                # add hyp to the best list
                if hyp.score >= overall_spot_score:
                    best_hyp_list_filtered.append(hyp)
                    current_word_in_ali = i
                    hyp_intersects = False
                    break
        # if hyp has not yet been added (end of sentence case)
        if hyp_intersects and hyp.score >= overall_spot_score:
            best_hyp_list_filtered.append(hyp)

    return best_hyp_list_filtered


def run_word_spotter(
    logprobs: np.ndarray,
    context_graph: ContextGraphCTC,
    asr_model,
    blank_idx: int = 0,
    beam_threshold: float = 5.0,
    cb_weight: float = 3.0,
    ctc_ali_token_weight: float = 0.5,
    keyword_threshold: float = -5.0,
    blank_threshold: float = 0.8,
    non_blank_threshold: float = 0.000001,
):
    """
    CTC-based Word Spotter for recognition of words from context biasing graph (paper link)
    The algorithm is based on the Token Passing Algorithm (TPA) and uses run, beam and state prunings.
    Blank and non-blank thresholds are used for preliminary hypotheses pruning.
    The algorithm is implemented in log semiring.

    Args:
        logprobs: CTC logprobs for one file [Time, Vocab+blank]
        context_graph: Context-Biasing graph
        blank_idx: blank index in ASR model
        asr_model: ASR model (ctc or hybrid-transducer-ctc)
        beam_threshold: threshold for beam pruning
        cb_weight: context biasing weight
        ctc_ali_token_weight: additional token weight for word-level ctc alignment
        keyword_threshold: auxiliary weight for pruning final hypotheses
        blank_threshold: blank threshold (probability) for preliminary hypotheses pruning
        non_blank_threshold: non-blank threshold (probability) for preliminary hypotheses pruning

    Returns:
        final list of spotted hypotheses WSHyp
    """

    start_state = context_graph.root
    active_tokens = []
    next_tokens = []
    spotted_words = []

    # move threshold probabilities to log space
    blank_threshold = np.log(blank_threshold)
    non_blank_threshold = np.log(non_blank_threshold)

    for frame in range(logprobs.shape[0]):
        # add an empty token (located in the graph root) at each new frame to start new word spotting
        active_tokens.append(Token(start_state, start_frame=frame))
        best_score = None
        for token in active_tokens:
            # skip token by the blank_threshold if empty token
            if token.state is context_graph.root and logprobs[frame][blank_idx] > blank_threshold:
                continue
            for transition_state in token.state.next:
                # skip non-blank token by the non_blank_threshold if empty token
                if token.state is context_graph.root and logprobs[frame][int(transition_state)] < non_blank_threshold:
                    continue
                # running beam pruning (start) - skips current token by score before Token class creations
                if transition_state != blank_idx:
                    # add cb_weight only for non-blank tokens
                    current_score = token.score + logprobs[frame][int(transition_state)].item() + cb_weight
                else:
                    current_score = token.score + logprobs[frame][int(transition_state)].item()
                if best_score is None:
                    best_score = current_score
                else:
                    if current_score < best_score - beam_threshold:
                        continue
                    elif current_score > best_score:
                        best_score = current_score
                # running beam pruning (end)

                new_token = Token(token.state.next[transition_state], current_score, token.start_frame)
                # add a word as spotted if token reached the end of word state in context graph:
                if new_token.state.is_end and new_token.score > keyword_threshold:
                    word = new_token.state.word
                    spotted_words.append(
                        WSHyp(word=word, score=new_token.score, start_frame=new_token.start_frame, end_frame=frame)
                    )
                    # check case when the current state is the last in the branch (only one self-loop transition)
                    if len(new_token.state.next) == 1:
                        if current_score == best_score:
                            best_score = None
                        continue
                next_tokens.append(new_token)
        # state and beam prunings:
        next_tokens = beam_pruning(next_tokens, beam_threshold)
        next_tokens = state_pruning(next_tokens)

        active_tokens = next_tokens
        next_tokens = []

    # find best hyps for spotted keywords (in case of hyps overlapping):
    best_hyp_list = find_best_hyps(spotted_words)

    # filter hyps according to word-level ctc alignment to avoid a high false accept rate
    ctc_word_alignment = get_ctc_word_alignment(
        logprobs, asr_model, token_weight=ctc_ali_token_weight, blank_idx=blank_idx
    )
    best_hyp_list = filter_wb_hyps(best_hyp_list, ctc_word_alignment)

    return best_hyp_list
