from torch.utils.data import Dataset
import json
import random
import settings
from transformers import AutoTokenizer


class TrainDataset(Dataset):
    def __init__(self):
        f = open("../data/train-claims-with-negatives.json", "r")
        self.dataset = json.load(f)
        f.close()
        f = open("../data/evidence.json", "r")
        self.evidences = json.load(f)
        f.close()
        self.claim_ids = list(self.dataset.keys())
        self.evidence_ids = list(self.evidences.keys())
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        data = self.dataset[self.claim_ids[idx]]
        query = data["claim_text"].lower()
        evidences = []
        for evidence_id in data["evidences"]:
            evidences.append(evidence_id)

        negative_evidences = data["negative_evidences"]
        return [query, evidences, negative_evidences]

    def collate_fn(self, batch):
        queries = []
        evidences = []
        labels = []
        negative_evidences = []
        for query, evidence, negative_evidence in batch:
            queries.append(query)
            evidences.extend(evidence)
            negative_evidences.extend(negative_evidence)
            labels.append(len(evidence))
        evidences.extend(negative_evidences)

        cur_evidences_num = len(evidences)
        if cur_evidences_num > settings.evidence_nums:
            evidences = evidences[:settings.evidence_nums]

        evidences_text = [self.evidences[evidence_id].lower() for evidence_id in evidences]
        while cur_evidences_num < settings.evidence_nums:
            evidence_id = random.choice(self.evidence_ids)
            while evidence_id in evidences:
                evidence_id = random.choice(self.evidence_ids)
            evidences.append(evidence_id)
            evidences_text.append(self.evidences[evidence_id].lower())
            cur_evidences_num += 1

        query_text = self.tokenizer(
            queries,
            max_length=128,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        evidence_text = self.tokenizer(
            evidences_text,
            max_length=128,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_tokened = dict()
        batch_tokened["query_input_ids"] = query_text.input_ids
        batch_tokened["evidence_input_ids"] = evidence_text.input_ids
        batch_tokened["query_attention_mask"] = query_text.attention_mask
        batch_tokened["evidence_attention_mask"] = evidence_text.attention_mask
        batch_tokened["labels"] = labels
        return batch_tokened


class EvidenceDataset(Dataset):
    def __init__(self):
        f = open("../data/evidence.json", "r")
        self.evidences = json.load(f)
        f.close()

        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.evidences_ids = list(self.evidences.keys())

    def __len__(self):
        return len(self.evidences_ids)

    def __getitem__(self, idx):
        evidences_id = self.evidences_ids[idx]
        evidence = self.evidences[evidences_id]
        return [evidences_id, evidence]

    def collate_fn(self, batch):
        evidences_ids = []
        evidences = []

        for evidences_id, evidence in batch:
            evidences_ids.append(evidences_id)
            evidences.append(evidence.lower())

        evidences_text = self.tokenizer(
            evidences,
            max_length=128,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_tokened = dict()
        batch_tokened["evidence_input_ids"] = evidences_text.input_ids
        batch_tokened["evidence_attention_mask"] = evidences_text.attention_mask
        batch_tokened["evidences_ids"] = evidences_ids
        return batch_tokened


class ValDataset(Dataset):
    def __init__(self, mode):
        if mode != "test":
            f = open("../data/{}-claims.json".format(mode), "r")
        else:
            f = open("../data/test-claims-unlabelled.json", "r")
        self.dataset = json.load(f)
        f.close()

        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.claim_ids = list(self.dataset.keys())
        self.mode = mode

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        data = self.dataset[self.claim_ids[idx]]
        text = data["claim_text"].lower()
        return [text, data, self.claim_ids[idx]]

    def collate_fn(self, batch):
        queries = []
        datas = []
        evidences = []
        claim_ids = []
        for query, data, claim_id in batch:
            queries.append(query)
            datas.append(data)
            if self.mode != "test":
                evidences.append(data["evidences"])
            claim_ids.append(claim_id)

        query_text = self.tokenizer(
            queries,
            max_length=128,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_tokened = dict()
        batch_tokened["query_input_ids"] = query_text.input_ids
        batch_tokened["query_attention_mask"] = query_text.attention_mask
        batch_tokened["datas"] = datas
        batch_tokened["claim_ids"] = claim_ids
        if self.mode != "test":
            batch_tokened["evidences"] = evidences
        return batch_tokened

