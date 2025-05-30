import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime
from loguru import logger
from tqdm import tqdm

def rerank_paper(candidate:list[ArxivPaper],corpus:list[dict],model:str='avsolatorio/GIST-small-Embedding-v0') -> list[ArxivPaper]:
    encoder = SentenceTransformer(model)
    #sort corpus by date, from newest to oldest
    corpus = sorted(corpus,key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),reverse=True)
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    logger.info("Encoding corpus abstracts...")
    logger.info("Corpus feature sample: {}".format(corpus[0]['data']['abstractNote']))
    corpus_feature = encoder.encode([paper['data']['abstractNote'] for paper in tqdm(corpus, desc="Corpus")])
    logger.info("Encoding candidate papers...")
    candidate_feature = encoder.encode([paper.summary for paper in tqdm(candidate, desc="Candidates")])
    
    sim = encoder.similarity(candidate_feature,corpus_feature) # [n_candidate, n_corpus]
    scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
    for s,c in zip(scores,candidate):
        c.score = s.item()
    # debug score
    for i,c in enumerate(corpus):
        logger.info(f"Corpus: {c['data']['title']}, Score: {sim[i].max()}, candidate: {candidate[sim[i].argmax()].title}")
    for i, c in enumerate(candidate):
        logger.info(f"Paper: {c.title}, Score: {c.score}, maxScore: {max(sim[i])}, zotero_paper: {corpus[sim[i].argmax()]['data']['title']}")

    candidate = keyword_score_update(candidate)
    candidate = sorted(candidate,key=lambda x: x.score,reverse=True)
    return candidate

def keyword_score_update(candidate:list[ArxivPaper]) -> list[ArxivPaper]:
    for c in candidate:
        if c.search_keyword:
            c.score = c.score + 2
    return candidate


