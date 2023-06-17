import json
import sys
import time
import numpy as np
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from scipy.linalg import norm
import settings
import gensim.downloader
import re


glove_vectors = gensim.downloader.load('glove-twitter-25')

class Word2Vec(object):
    def __init__(self, docs):
        self.docs = docs
        self.doc_num = len(docs)
        self.vocab = set([word for doc in self.docs for word in doc])

    def sentence_vector(self, doc):
        v = np.zeros(25)
        count = 0
        for word in doc:
            if glove_vectors.has_index_for(word):
                v += glove_vectors[word]
                count += 1
        if count != 0:
            v /= count
        return v

    def score_cos(self, v1, v2):
        if norm(v1)*norm(v2) != 0:
            return np.dot(v1, v2) / (norm(v1) * norm(v2))
        else:
            return 0

    def score_all(self, sequence):
        all_progress = len(self.docs)
        count_progress = 0
        scores = []
        v_query = self.sentence_vector(sequence)
        for doc in self.docs:

            if len(doc)>0:
                v_doc = self.sentence_vector(doc)
                scores.append(self.score_cos(v_query, v_doc))
            else:
                scores.append(0)
            count_progress += 1
            progress = count_progress / all_progress * 100
            progress = round(progress, 1)
            print("\r", end="")
            print('progress：{}%'.format(progress), "▋" * (int(round(progress)) // 2), end="")
            sys.stdout.flush()
            time.sleep(0.00001)
        print('')
        return scores


if __name__ == '__main__':
    f = open("../data/test-claims-unlabelled.json", "r")
    dataset = json.load(f)
    claim_ids = list(dataset.keys())
    f = open("../data/evidence.json", "r")
    evidences = json.load(f)
    evidences_ids = list(evidences.keys())
    tt = TweetTokenizer()
    stopwords = set(stopwords.words('english'))  # note: stopwords are all in lowercase

    processed_evidences = []
    for i in range(len(evidences)):
        cur_evi_text = evidences[evidences_ids[i]]
        # text = data_utils.process(cur_evi_text)
        cur_token_tweet = tt.tokenize(cur_evi_text)
        cur_lower_token_tweet = [token.lower() for token in cur_token_tweet]
        english_tweet = [word for word in cur_lower_token_tweet if re.search('[a-zA-Z]', word)]
        removed_tweet = [word for word in english_tweet if word not in stopwords]
        processed_evidences.append(removed_tweet)
    model = Word2Vec(processed_evidences)
    k = settings.retrieval_num

    for i in range(len(dataset)):
        claim_text = dataset[claim_ids[i]]['claim_text']
        cur_token_tweet = tt.tokenize(claim_text)
        cur_lower_token_tweet = [token.lower() for token in cur_token_tweet]
        english_tweet = [word for word in cur_lower_token_tweet if re.search('[a-zA-Z]', word)]
        removed_tweet = [word for word in english_tweet if word not in stopwords]

        scores = model.score_all(removed_tweet)
        topk_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        for j in topk_idx:
            evidences = []
            evidences.append(evidences_ids[j])
            dataset[claim_ids[i]]['evidences'] = evidences

    fout = open("data/retrieval-test-claims.json", 'w')
    json.dump(dataset, fout)
    fout.close()

