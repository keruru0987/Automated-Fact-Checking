import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModel
from data_utils import  ValDataset, EvidenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import settings


# settings.model_pt = "ret"  # modify if using new path to save model
tok = AutoTokenizer.from_pretrained(settings.model_name)
test_set = ValDataset("test")
evidence_set = EvidenceDataset()

dataloader = DataLoader(test_set, batch_size=settings.batch_size, shuffle=False, num_workers=4,
                        collate_fn=test_set.collate_fn)
evidence_dataloader = DataLoader(evidence_set, batch_size=128, shuffle=False, num_workers=4,
                                 collate_fn=evidence_set.collate_fn)
# build models
query_model = AutoModel.from_pretrained(settings.model_name)
evidence_model = AutoModel.from_pretrained(settings.model_name)

query_model.load_state_dict(torch.load(os.path.join("./cache", settings.model_path, "query_model.bin")))
evidence_model.load_state_dict(torch.load(os.path.join("./cache", settings.model_path, "evidence_model.bin")))
query_model.cuda()
evidence_model.cuda()
query_model.eval()
evidence_model.eval()

# get evidence embedding and normalise

evidence_embeddings = torch.load(os.path.join("./cache", settings.model_path, "evidence_embeddings"))
evidence_ids = torch.load(os.path.join("./cache", settings.model_path, "evidence_ids"))

out_data = {}
for batch in tqdm(dataloader):
    for n in batch.keys():
        if n in ["query_input_ids", "evidence_input_ids", "query_attention_mask", "evidence_attention_mask"]:
            batch[n] = batch[n].cuda()
    query_last = query_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
    query_embedding = query_last[:, 0, :]
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).cpu()
    scores = torch.mm(query_embedding, evidence_embeddings)
    topk_ids = torch.topk(scores, k=settings.retrieval_num, dim=1).indices.tolist()
    for idx, data in enumerate(batch["datas"]):
        data["evidences"] = [evidence_ids[i] for i in topk_ids[idx]]
        out_data[batch["claim_ids"][idx]] = data
fout = open("../data/retrieval-test-claims.json", 'w')
json.dump(out_data, fout)
fout.close()


dev_set = ValDataset("dev")

dataloader = DataLoader(dev_set, batch_size=settings.batch_size, shuffle=False, num_workers=4,
                        collate_fn=dev_set.collate_fn)
out_data = {}
f = []
for batch in tqdm(dataloader):
    for n in batch.keys():
        if n in ["query_input_ids", "evidence_input_ids", "query_attention_mask", "evidence_attention_mask"]:
            batch[n] = batch[n].cuda()
    query_last = query_model(input_ids=batch["query_input_ids"],
                               attention_mask=batch["query_attention_mask"]).last_hidden_state
    query_embedding = query_last[:, 0, :]
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).cpu()
    scores = torch.mm(query_embedding, evidence_embeddings)
    topk_ids = torch.topk(scores, k=settings.retrieval_num, dim=1).indices.tolist()
    for idx, data in enumerate(batch["datas"]):
        pred_evidences = [evidence_ids[i] for i in topk_ids[idx]]
        data["evidences"] = pred_evidences
        out_data[batch["claim_ids"][idx]] = data
        evidence_correct = 0
        for evidence_id in batch["evidences"][idx]:
            if evidence_id in pred_evidences:
                evidence_correct += 1
        if evidence_correct > 0:
            evidence_recall = float(evidence_correct) / len(batch["evidences"][idx])
            evidence_precision = float(evidence_correct) / len(pred_evidences)
            evidence_fscore = (2 * evidence_precision * evidence_recall) / (evidence_precision + evidence_recall)
        else:
            evidence_fscore = 0
        f.append(evidence_fscore)
fscore = np.mean(f)
print('the fscore in dev is: ', fscore)
fout = open("../data/retrieval-dev-claims.json", 'w')
json.dump(out_data, fout)
fout.close()

# nohup python -u retrieval.py >test.out 2>&1 &
