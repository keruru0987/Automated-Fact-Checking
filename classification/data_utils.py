from torch.utils.data import Dataset
import json
import torch
from transformers import AutoTokenizer
import settings


class ClsDataset(Dataset):
    def __init__(self, mode):
        if mode != "test":
            f = open("../data/{}-claims.json".format(mode), "r")
        else:
            f = open("../data/retrieval-test-claims.json", "r")
        self.dataset = json.load(f)
        f.close()
        f = open("../data/evidence.json", "r")
        self.evidences = json.load(f)
        f.close()

        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_type)
        self.claim_ids = list(self.dataset.keys())
        self.mode = mode

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        data = self.dataset[self.claim_ids[idx]]
        input_text = [data["claim_text"].lower()]
        for evidence_id in data["evidences"]:
            input_text.append(self.evidences[evidence_id].lower())
        input_text = self.tokenizer.sep_token.join(input_text)
        if self.mode != "test":
            label = settings.label2ids[data["claim_label"]]
            return [input_text, label, data, self.claim_ids[idx]]
        else:
            return [input_text, data, self.claim_ids[idx]]

    def collate_fn(self, batch):
        batch_tokened = dict()
        input_texts = []
        datas = []
        claim_ids = []
        if self.mode != "test":
            labels = []
            for input_text, label, data, claim_id in batch:
                input_texts.append(input_text)
                datas.append(data)
                claim_ids.append(claim_id)
                labels.append(label)
            batch_tokened["label"] = torch.LongTensor(labels)
        else:
            for input_text, data, claim_id in batch:
                input_texts.append(input_text)
                datas.append(data)
                claim_ids.append(claim_id)

        src_text = self.tokenizer(
            input_texts,
            max_length=1024,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_tokened["input_ids"] = src_text.input_ids
        batch_tokened["attention_mask"] = src_text.attention_mask
        batch_tokened["datas"] = datas
        batch_tokened["claim_ids"] = claim_ids

        return batch_tokened
