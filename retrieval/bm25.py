import json
import sys
import time
import numpy as np
from collections import Counter

from nltk import TweetTokenizer
from nltk.corpus import stopwords

import settings
import re


class BM25(object):
    def __init__(self, docs):
        self.docs = docs
        self.doc_num = len(docs)
        self.vocab = set([word for doc in self.docs for word in doc])
        self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.doc_num
        self.k1 = 1.5
        self.b = 0.75

    def idf(self, word):
        if word not in self.vocab:
            word_idf = 0
        else:
            qn = {}
            for doc in self.docs:
                if word in doc:
                    if word in qn:
                        qn[word] += 1
                    else:
                        qn[word] = 1
                else:
                    continue
            word_idf = np.log((self.doc_num - qn[word] + 0.5) / (qn[word] + 0.5))
        return word_idf

    def score(self, word):
        score_list = []
        for index, doc in enumerate(self.docs):
            word_count = Counter(doc)
            if word in word_count.keys():
                f = (word_count[word]+0.0) / len(doc)
            else:
                f = 0.0
            r_score = (f*(self.k1+1)) / (f+self.k1*(1-self.b+self.b*len(doc)/self.avgdl))
            score_list.append(self.idf(word) * r_score)
        return score_list

    def score_all(self, sequence):
        all_progress = len(sequence)
        count_progress = 0

        score = []
        for word in sequence:
            score.append(self.score(word))
            # print('the round', count, '/', len(sequence), 'complete')
            count_progress += 1
            progress = count_progress / all_progress * 100
            progress = round(progress, 1)
            print("\r", end="")
            print('进度：{}%'.format(progress), "▋" * (int(round(progress)) // 2), end="")
            sys.stdout.flush()
            time.sleep(0.00001)
        sum_score = np.sum(score, axis=0)

        if len(sequence) != 0:
            sum_score /= len(sequence)

        print('')
        return sum_score.tolist()


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
    model = BM25(processed_evidences)
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
