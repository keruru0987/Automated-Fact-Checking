import torch
import os
from transformers import AutoTokenizer
from data_utils import ClsDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from train import CLSModel
import settings

# load data

tok = AutoTokenizer.from_pretrained(settings.model_type)
test_set = ClsDataset("test")

dataloader = DataLoader(test_set, batch_size=settings.batch_size, shuffle=False, num_workers=4,
                        collate_fn=test_set.collate_fn)
# build models
model = CLSModel(settings.model_type)
model.load_state_dict(torch.load(os.path.join("./cache", settings.model_pt, "best_ckpt.bin")))

model.cuda()
model.eval()

out_data = {}
for batch in tqdm(dataloader):
    for n in batch.keys():
        if n in ["input_ids", "attention_mask", "label"]:
            batch[n] = batch[n].cuda()
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    predict_labels = logits.argmax(-1).tolist()
    idx = 0
    for data, predict_label in zip(batch["datas"], predict_labels):
        data["claim_label"] = settings.ids2label[predict_label]
        out_data[batch["claim_ids"][idx]] = data
        idx += 1
fout = open("test-claims-predictions.json", 'w')
json.dump(out_data, fout)
fout.close()


# nohup python -u predict.py >train.out 2>&1 &