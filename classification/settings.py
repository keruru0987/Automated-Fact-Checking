seed = 17
label2ids = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
ids2label = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
model_type = "roberta-large"
model_pt = "cls"
using_pre_model = 1
batch_size = 4
epoch_times = 10
accumulate_step = 4
eval_frequency = 20
learning_rate = 1e-7




