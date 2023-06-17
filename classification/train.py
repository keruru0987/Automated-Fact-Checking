import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from transformers import AutoModel
from data_utils import ClsDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import settings


class CLSModel(nn.Module):
    def __init__(self, pre_encoder):
        super(CLSModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(pre_encoder)
        hidden_size = self.encoder.config.hidden_size
        self.cls = nn.Linear(hidden_size, 4)

    def forward(self, input_ids, attention_mask):
        texts_embedding = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # first token
        texts_embedding = texts_embedding[:, 0, :]
        logits = self.cls(texts_embedding)
        return logits


def validate(model):
    val_set = ClsDataset("dev")
    val_dataloader = DataLoader(val_set, batch_size=settings.batch_size, shuffle=False, num_workers=4,
                                collate_fn=val_set.collate_fn)
    model.eval()
    cnt = 0.
    correct_cnt = 0.
    for batch in tqdm(val_dataloader):
        for n in batch.keys():
            if n in ["input_ids", "attention_mask", "label"]:
                batch[n] = batch[n].cuda()
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predict_labels = logits.argmax(-1)
        result = predict_labels == batch["label"]
        correct_cnt += result.sum().item()
        cnt += predict_labels.size(0)
    acc = correct_cnt / cnt

    print("evaluation accuracy: %.3f" % acc)

    model.train()
    return acc


def run():
    # task initialization
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)

    train_set = ClsDataset("train")
    dataloader = DataLoader(train_set, batch_size=settings.batch_size, shuffle=True, num_workers=4, collate_fn=train_set.collate_fn)

    # build models
    model = CLSModel(settings.model_type)

    if settings.using_pre_model == 1:
        model.load_state_dict(torch.load(os.path.join("./cache", settings.model_pt, "best_ckpt.bin")))
    model.cuda()
    model.train()

    save_dir = f"./cache/{settings.model_pt}"
    os.makedirs(save_dir, exist_ok=True)

    ce_fn = nn.CrossEntropyLoss()
    s_optimizer = optim.Adam(model.parameters())

    # keep lr fixed
    for param_group in s_optimizer.param_groups:
        param_group['lr'] = settings.learning_rate

    # start training
    s_optimizer.zero_grad()
    step_cnt = 0
    optim_cnt = 0
    avg_loss = 0
    maximum_acc = 0

    for epoch in range(settings.epoch_times):
        epoch_cnt = 0
        for (i, batch) in enumerate(tqdm(dataloader)):
            for n in batch.keys():
                if n in ["input_ids", "attention_mask", "label"]:
                    batch[n] = batch[n].cuda()
            step_cnt += 1
            # forward pass
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

            loss = ce_fn(logits, batch["label"])
            loss = loss / settings.accumulate_step
            loss.backward()

            avg_loss += loss.item()
            if step_cnt == settings.accumulate_step:
                # updating
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                step_cnt = 0
                epoch_cnt += 1
                optim_cnt += 1

                s_optimizer.step()
                s_optimizer.zero_grad()

                print("epoch: %d, epoch_step: %d, avg loss: %.6f" % (epoch + 1, epoch_cnt, avg_loss))
                avg_loss = 0

            del loss, logits

            if optim_cnt % settings.eval_frequency == 0 and optim_cnt != 0 and step_cnt == 0:
                print("Evaluate:\n")
                acc = validate(model)
                if acc > maximum_acc:
                    maximum_acc = acc
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_ckpt.bin"))
                    print("maximum_acc", acc)


if __name__ == "__main__":
    # nohup python -u train.py >train.out 2>&1 &
    settings.model_pt = "cls"  # modify if using new path to save model
    settings.using_pre_model = 0  # set 1 if using previous ckpt
    run()
