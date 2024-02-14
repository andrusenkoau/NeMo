---
title: CTC-WS
author: [Andrei Andrusenko, Aleksandr Laptev, Vladimir Bataev, Vitaly Lavrukhin, Boris Ginsburg]
author_gh_user: [andrusenkoau, GNroy, artbataev, vsl9, borisgin]
readtime: 10
date: 2024-02-16

# Optional: Categories
categories:  
  - Announcements

# Optional: OpenGraph metadata
# og_title: NVIDIA NeMo CTCWS
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Fast context-biasing for CTC and Transducer ASR models with CTC-based Words Spotter
---

<!-- #region -->
# A new fast context-biasing method for CTC and RNNT models with CTC-based Word Spotter

Improving the recognition of rare and new words is essential for contextualized Automatic Speech Recognition (ASR) systems. Most context-biasing methods are based on ASR model modifications or decoding in beam-search mode that require model retraining or resource-intensive decoding. This post explains how to use a fast and accurate context-biasing method by CTC-based Word Spotter (CTC-WS) for CTC and Transducer (RNNT) ASR models in NeMo.

<figure markdown>
  <img src="https://github.com/NVIDIA/NeMo/releases/download/v1.22.0/asset-post-v1.22.0-ctcws_scheme_2.png" alt="CTC-WS" style="width: 60%;" height="auto"> <!-- Adjust the width as needed -->
  <figcaption><b>Figure 1.</b> <i> High-level representation of the proposed context-biasing method with CTC-WS in case of CTC model. Detected words (gpu, nvidia, cuda) are compared with words from the greedy CTC results in the overlapping intervals according to the accumulated scores to prevent false accept replacement. </i></figcaption>
</figure>

<!-- more -->

## Context-biasing intro

In Automatic Speech Recognition (ASR), there is a common challenge related to identifying words that have limited representation or are entirely absent in the training data. This issue becomes especially pronounced as new names and terms continually emerge in our fast-evolving world. Users expect ASR systems to adapt and recognize these novel words seamlessly. To address this, researchers have devised context-biasing methods based on having a predefined list of words and phrases to enhance their recognition accuracy.

One avenue within context-biasing methods is the deep fusion approach. These methods require integration into the ASR model and its training process which has a drawback – they demand substantial computational resources and time for model training.

Alternatively, there are methods employing a shallow fusion approach. Unlike deep fusion, these techniques focus solely on adjusting the decoding process. During decoding, hypotheses are rescored based on their presence in the context-biasing list or an external language model. While this approach reduces the computational burden compared to deep fusion, it still poses challenges, particularly with beam-search decoding. This challenge escalates for models with extensive vocabularies and context-biasing lists.

Moreover, for models like the Transducer (RNNT) model, which involves multiple calculations of Decoder (Prediction) and Joint networks during beam-search decoding, the computational load becomes even more burdensome. Furthermore, the effectiveness of context-biasing recognition is contingent upon the model's initial predictions. In scenarios involving rare or newly introduced words, the model may lack a hypothesis for the desired word from the context-biasing list, thereby preventing the improvement of its recognition.

## CTC-WS

The NVIDIA NeMo team introduces a novel fast context-biasing technique employing a CTC-based Word Spotter (CTC-WS). This method involves decoding CTC log probabilities using a context graph constructed from words and phrases in the context-biasing list. Detected context-biasing candidates (along with their scores and time intervals) are compared against the scores of words from the greedy CTC decoding results to enhance recognition accuracy and mitigate false acceptances of context-biasing (see Figure 2).

<figure markdown>
  <img src="https://github.com/NVIDIA/NeMo/releases/download/v1.22.0/asset-post-v1.22.0-ctcws_scheme_1.png" alt="CTC-WS" style="width: 65%;" align="center"> <!-- Adjust the width as needed -->
  <figcaption><b>Figure 2.</b> <i> Scheme of the context-biasing method with CTC-based Word Spotter. CTC-WS uses CTC log probabilities to detect context-biasing candidates. Obtained candidates are filtered by CTC word alignment and then merged with CTC or RNN-T word alignment to get the final text result. </i></figcaption>
</figure>


Utilizing a Hybrid Transducer-CTC model (a shared encoder trained jointly with CTC and Transducer output heads) enables the integration of the CTC-WS method into the Transducer model. Context-biasing candidates identified by CTC-WS are further refined based on their scores compared to greedy CTC predictions before merging with greedy Transducer results.

The CTC-WS technique facilitates the utilization of pretrained NeMo models (CTC or Hybrid Transducer-CTC) for context-biasing recognition without necessitating model retraining. This approach demonstrates promising accuracy results for context-biasing with minimal additional time and computational resources in comparison with other shallow fusion approaches:
  

<figure markdown>
  <img src="https://github.com/NVIDIA/NeMo/releases/download/v1.22.0/asset-post-v1.22.0-ctcws_results.png" alt="CTC-WS-results" style="width: 40%;"> <!-- Adjust the width as needed -->
</figure>

See more details in the CTC-WS paper – link.

## How to use CTC-WS

Check out our tutorial for context-biasing via CTC-WS method for CTC and Transducer models – tutorial link.

<!-- #endregion -->

```python

```
