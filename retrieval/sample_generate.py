import json
import torch
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re
from bm25 import BM25
from vsm import VSM
import settings
from data_utils import ValDataset
from torch.utils.data import DataLoader


def generate_neg_samples(query_model, evidence_embeddings, evidence_ids):
    train_set = ValDataset("train")
    dataloader = DataLoader(train_set, batch_size=settings.batch_size, shuffle=False, num_workers=4, collate_fn=train_set.collate_fn)
    query_model.eval()
    out_data = {}
    for batch in tqdm(dataloader):
        for n in batch.keys():
            if n in ["query_input_ids", "evidence_input_ids", "query_attention_mask", "evidence_attention_mask"]:
                batch[n] = batch[n].cuda()
        query_last = query_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
        query_embedding = query_last[:, 0, :]
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).cpu()
        scores = torch.mm(query_embedding, evidence_embeddings)
        topk_ids = torch.topk(scores, k=settings.generate_neg_num, dim=1).indices.tolist()
        for idx, data in enumerate(batch["datas"]):
            negative_evidences = []
            for i in topk_ids[idx]:
                if evidence_ids[i] not in batch["evidences"][idx]:
                    negative_evidences.append(evidence_ids[i])
            data["negative_evidences"] = negative_evidences
            out_data[batch["claim_ids"][idx]] = data
    fout = open("../data/train-claims-with-negatives.json", 'w')
    json.dump(out_data, fout)
    fout.close()


k = settings.generate_neg_num
# choose one from three
model_name = 'bm25'
# model_name = 'vsm'
# model_name = 'word2vec'

f = open("../data/train-claims.json", "r")
dataset = json.load(f)
claim_ids = list(dataset.keys())
f = open("../data/evidence.json", "r")
evidences = json.load(f)
evidences_ids = list(evidences.keys())

print(1)

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

print(2)

if model_name == 'bm25':
    select_model = BM25(processed_evidences)
elif model_name == 'vsm':
    select_model = VSM(processed_evidences)
elif model_name == 'word2vec':
    select_model = BM25(processed_evidences)
else:
    raise Exception('no such model found')

for i in range(len(dataset)):
    claim_text = dataset[claim_ids[i]]['claim_text']
    cur_token_tweet = tt.tokenize(claim_text)
    cur_lower_token_tweet = [token.lower() for token in cur_token_tweet]
    english_tweet = [word for word in cur_lower_token_tweet if re.search('[a-zA-Z]', word)]
    removed_tweet = [word for word in english_tweet if word not in stopwords]

    scores = select_model.score_all(removed_tweet)
    topk_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    for j in topk_idx:
        negative_evidences = []
        if evidences_ids[j] not in dataset[claim_ids[i]]['evidences']:
            negative_evidences.append(evidences_ids[j])
        dataset[claim_ids[i]]['negative_evidences'] = negative_evidences

fout = open("data/train-claims-with-negatives.json", 'w')
json.dump(dataset, fout)
fout.close()

