import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk import ngrams
import json
from bert_score import BERTScorer

def bert_similarity_score(generated_answer, correct_answer):
    # Initialize the BERT scorer
    scorer = BERTScorer(lang="en", model_type="bert-base-uncased")
    f1s= []
    for idx in range(len(generated_answer)):
        # Calculate BERT similarity scores
        gen = generated_answer[idx]
        gt = correct_answer[idx]
        P, R, F1 = scorer.score([gen], [gt])
        f1s.append(F1.item())
    return np.mean(f1s)  # Extract the F1 score

def compute_language_model_scores(strings, target):

    # "ref": [ "dotted T2WI hyperintensity on left and right frontal lobe.",
    #        "dotted T2WI hyperintensity on left and right parietal lobe."]
    # "gen": [ "dotted T2WI hyperintensity on left and right frontal lobe.",
    #        "dotted T2WI hyperintensity on left and right parietal lobe."]
    # "bleu": 0.1388888888888889,
    # "rouge": 0.8333333333333334

    bleus = []
    rouges = []

    for idx in range(len(strings)):
        
        s = strings[idx]
        t = target[idx]

        bleus.append(sentence_bleu([t.lower().strip().split()], s.lower().strip().split(), weights=(1, 0, 0, 0)))
        # print("sentence_bleu", bleus[-1])

        # strings example ['sbc','def','hgk']
        # target example [['sds','sd','cvb'],['a','b','c']]
        # if strings is very similar to one of the target example, then the score will be high
        # weights = (1,0,0,0) means 1-gram weight 1, 2-gram, 3-gram, 4-gram weight 0
        
        summary_tokens = s.lower().strip().split()
        reference_tokens = t.lower().strip().split()

        # print("strings",summary_tokens)
        # print("target",reference_tokens)

        summary_ngrams = set(ngrams(summary_tokens, 1))
        reference_ngrams = set(ngrams(reference_tokens, 1))
        
        overlap = len(summary_ngrams.intersection(reference_ngrams))

        if len(reference_ngrams) == 0:
            rouges.append(1)
        else:
            rouges.append(overlap / len(reference_ngrams))

        # print("rouge",rouges[-1])

    return rouges, bleus