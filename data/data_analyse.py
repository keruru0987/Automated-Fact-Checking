import json

f = open("train-claims.json", "r")
dataset = json.load(f)
f.close()

print('len dataset train:', len(dataset))

gold_retrieval_nums = []
text_lens = []
for key in dataset.keys():
    gold_retrieval_nums.append(len(dataset[key]["evidences"]))
    text_lens.append(len(dataset[key]["claim_text"].split()))

print(min(gold_retrieval_nums), max(gold_retrieval_nums), sum(gold_retrieval_nums) / len(gold_retrieval_nums))
print(min(text_lens), max(text_lens), sum(text_lens) / len(text_lens))


f = open("dev-claims.json", "r")
dataset = json.load(f)
f.close()

print('len dev train:', len(dataset))

gold_retrieval_nums = []
text_lens = []
for key in dataset.keys():
    gold_retrieval_nums.append(len(dataset[key]["evidences"]))
    text_lens.append(len(dataset[key]["claim_text"].split()))

print(min(gold_retrieval_nums), max(gold_retrieval_nums), sum(gold_retrieval_nums) / len(gold_retrieval_nums))
print(min(text_lens), max(text_lens), sum(text_lens) / len(text_lens))


f = open("evidence.json", "r")
dataset = json.load(f)
f.close()

print('len dataset evidence:', len(dataset))

text_lens = []
max_128 = 0
for key in dataset.keys():
    l = len(dataset[key].split())
    if l > 128:
        max_128 += 1
    text_lens.append(l)
print(max_128)
print(min(text_lens), max(text_lens), sum(text_lens) / len(text_lens))

