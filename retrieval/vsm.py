import json
import sys
import time
import math
import pandas as pd
from nltk import TweetTokenizer
from nltk.corpus import stopwords
import re

import settings


class VSM(object):
    def __init__(self, docs):
        self.docs = docs
        self.doc_num = len(docs)
        self.vocab = set([word for doc in self.docs for word in doc])

    def computeTF(self, vocab, doc):
        tf = dict.fromkeys(vocab, 0)
        for word in doc:
            tf[word] += 1
        return tf

    def computeIDF(self, tfList):
        idfDict = dict.fromkeys(tfList[0], 0)
        N = len(tfList)
        for tf in tfList:
            for word, count in tf.items():
                if count > 0:
                    idfDict[word] += 1
        for word, Ni in idfDict.items():
            idfDict[word] = math.log10(N / Ni)
        return idfDict

    def computeTFIDF(self, tf, idfs):
        tfidf = {}
        for word, tfval in tf.items():
            tfidf[word] = tfval * idfs[word]
        return tfidf

    def score_all(self, sequence):
        all_progress = len(self.docs)*3
        count_progress = 0
        tf_list = []
        for doc in self.docs:
            tf = self.computeTF(self.vocab, doc)
            tf_list.append(tf)
            count_progress += 1
            progress = count_progress / all_progress * 100
            progress = round(progress, 1)
            print("\r", end="")
            print('progress：{}%'.format(progress), "▋" * (int(round(progress)) // 2), end="")
            sys.stdout.flush()
            time.sleep(0.00001)

        idfs = self.computeIDF(tf_list)
        tf_idf_list = []
        for tf in tf_list:
            tf_idf = self.computeTFIDF(tf, idfs)
            tf_idf_list.append(tf_idf)
            count_progress += 1
            progress = count_progress / all_progress * 100
            progress = round(progress, 1)
            print("\r", end="")
            print('progress：{}%'.format(progress), "▋" * (int(round(progress)) // 2), end="")
            sys.stdout.flush()
            time.sleep(0.00001)

        Dvector = pd.DataFrame([tfidf for tfidf in tf_idf_list])

        query = []
        for word in sequence:
            if word in self.vocab:
                query.append(word)
            else:
                continue
        tf = self.computeTF(self.vocab, query)
        Q_tf_idf = self.computeTFIDF(tf, idfs)

        scores = []
        for vector in Dvector.to_dict(orient='records'):
            score = 0
            for k in Q_tf_idf:
                if k in vector:
                    score += Q_tf_idf[k]*vector[k]
            scores.append(score)
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
    model = VSM(processed_evidences)
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