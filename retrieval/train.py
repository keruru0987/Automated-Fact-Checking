import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from transformers import AutoModel
from data_utils import TrainDataset, ValDataset, EvidenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import settings
from sample_generate import generate_neg_samples


def get_evidence_embeddings(evidence_dataloader, evidence_model):
    evidence_model.eval()
    evidence_ids = []
    evidence_embeddings = []
    for batch in tqdm(evidence_dataloader):
        for n in batch.keys():
            if n in ["query_input_ids", "evidence_input_ids", "query_attention_mask", "evidence_attention_mask"]:
                batch[n] = batch[n].cuda()
        evidence_last = evidence_model(input_ids=batch["evidence_input_ids"], attention_mask=batch["evidence_attention_mask"]).last_hidden_state
        evidence_embedding = evidence_last[:, 0, :].detach()
        evidence_embedding_cpu = torch.nn.functional.normalize(evidence_embedding, p=2, dim=1).cpu()
        del evidence_embedding, evidence_last
        evidence_embeddings.append(evidence_embedding_cpu)
        evidence_ids.extend(batch["evidences_ids"])
    evidence_embeddings = torch.cat(evidence_embeddings, dim=0).t()
    evidence_model.train()
    return evidence_embeddings, evidence_ids


def validate(evidence_embeddings, evidence_ids, query_model):
    val_set = ValDataset("dev")
    val_dataloader = DataLoader(val_set, batch_size=settings.batch_size, shuffle=False, num_workers=4,
                                collate_fn=val_set.collate_fn)
    query_model.eval()
    f = []
    for batch in tqdm(val_dataloader):
        for n in batch.keys():
            if n in ["query_input_ids", "evidence_input_ids", "query_attention_mask", "evidence_attention_mask"]:
                batch[n] = batch[n].cuda()
        query_last = query_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
        query_embedding = query_last[:, 0, :]
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).cpu()
        scores = torch.mm(query_embedding, evidence_embeddings)
        topk_ids = torch.topk(scores, k=settings.retrieval_num, dim=1).indices.tolist()

        for idx, data in enumerate(batch["datas"]):
            evidence_correct = 0
            pred_evidences = [evidence_ids[i] for i in topk_ids[idx]]
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
    print("Evidence Retrieval F-score: ", fscore)
    query_model.train()
    return fscore


def run():
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)

    query_model = AutoModel.from_pretrained(settings.model_name)
    evidence_model = AutoModel.from_pretrained(settings.model_name)

    if settings.using_pre_model == 1:
        query_model.load_state_dict(torch.load(os.path.join("./cache", settings.model_path, "query_model.bin")))
        evidence_model.load_state_dict(torch.load(os.path.join("./cache", settings.model_path, "evidence_model.bin")))

    query_model.cuda()
    evidence_model.cuda()
    query_model.eval()
    evidence_model.eval()

    date = datetime.now().strftime("%y-%m-%d")
    save_dir = f"./cache/{date}"
    os.makedirs(save_dir, exist_ok=True)

    query_optimizer = optim.Adam(query_model.parameters())
    evidence_optimizer = optim.Adam(evidence_model.parameters())

    # lr
    for param_group in query_optimizer.param_groups:
        param_group['lr'] = settings.learning_rate
    for param_group in evidence_optimizer.param_groups:
        param_group['lr'] = settings.learning_rate

    # start training
    query_optimizer.zero_grad()
    evidence_optimizer.zero_grad()

    evidence_set = EvidenceDataset()
    evidence_dataloader = DataLoader(evidence_set, batch_size=128, shuffle=False, num_workers=4, collate_fn=evidence_set.collate_fn)

    print("Pre Evaluate:\n")
    cur_best_evidence_embeddings, cur_best_evidence_ids = get_evidence_embeddings(evidence_dataloader, evidence_model)
    f_score = validate(cur_best_evidence_embeddings, cur_best_evidence_ids, query_model)
    maximum_f_score = f_score

    print("pre f_score: ", f_score)

    step_cnt = 0
    optim_cnt = 0
    acc_loss = 0

    for epoch in range(settings.epoch_times):
        epoch_cnt = 0

        generate_neg_samples(query_model, cur_best_evidence_embeddings, cur_best_evidence_ids)
        train_set = TrainDataset()
        dataloader = DataLoader(train_set, batch_size=settings.batch_size, shuffle=True, num_workers=4, collate_fn=train_set.collate_fn)

        for (i, batch) in enumerate(tqdm(dataloader)):
            for n in batch.keys():
                if n in ["query_input_ids", "evidence_input_ids", "query_attention_mask", "evidence_attention_mask"]:
                    batch[n] = batch[n].cuda()
            step_cnt += 1

            query_embeddings = query_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
            query_embeddings = query_embeddings[:, 0, :]
            query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)

            evidence_embeddings = evidence_model(input_ids=batch["evidence_input_ids"], attention_mask=batch["evidence_attention_mask"]).last_hidden_state
            evidence_embeddings = evidence_embeddings[:, 0, :]
            evidence_embeddings = torch.nn.functional.normalize(evidence_embeddings, p=2, dim=1)

            cos_sims = torch.mm(query_embeddings, evidence_embeddings.t())
            scores = - torch.nn.functional.log_softmax(cos_sims / 0.05, dim=1)

            loss = []
            start_idx = 0
            for idx, label in enumerate(batch["labels"]):
                end_idx = start_idx + label
                cur_loss = torch.mean(scores[idx, start_idx:end_idx])
                loss.append(cur_loss)
                start_idx = end_idx

            loss = torch.stack(loss).mean()
            loss = loss / settings.accumulate_step
            loss.backward()

            acc_loss += loss.item()
            if step_cnt == settings.accumulate_step:
                # updating
                nn.utils.clip_grad_norm_(query_model.parameters(), 1)
                nn.utils.clip_grad_norm_(evidence_model.parameters(), 1)

                step_cnt = 0
                epoch_cnt += 1
                optim_cnt += 1

                query_optimizer.step()
                evidence_optimizer.step()
                query_optimizer.zero_grad()
                evidence_optimizer.zero_grad()

                print("epoch: %d, epoch_step: %d, avg loss: %.6f" % (epoch + 1, epoch_cnt, acc_loss))
                acc_loss = 0

                if optim_cnt % settings.eval_frequency == 0:
                    # evaluate the model as a scorer
                    print("\nEvaluate:\n")
                    evidence_embeddings, evidence_ids = get_evidence_embeddings(evidence_dataloader, evidence_model)
                    f_score = validate(evidence_embeddings, evidence_ids, query_model)
                    print('fscore: ', f_score)

                    if f_score > maximum_f_score:
                        print("max f_score: ", f_score)
                        maximum_f_score = f_score
                        cur_best_evidence_embeddings = evidence_embeddings
                        cur_best_evidence_ids = evidence_ids
                        torch.save(query_model.state_dict(), os.path.join(save_dir, "query_model.bin"))
                        torch.save(evidence_model.state_dict(), os.path.join(save_dir, "evidence_model.bin"))
                        torch.save(cur_best_evidence_embeddings, os.path.join(save_dir, "evidence_embeddings"))
                        torch.save(cur_best_evidence_ids, os.path.join(save_dir, "evidence_ids"))

            del loss, cos_sims, query_embeddings, evidence_embeddings


if __name__ == "__main__":
    # nohup python -u train.py >train.out 2>&1 &  # shell command
    settings.model_path = "ret"  # modify if using new path to save model
    settings.using_pre_model = 0  # if continue from checkpoint, set 1
    run()



