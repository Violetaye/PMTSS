import argparse
import logging
import os
import sys
from TrainSetting import MyDataSet, TrainArgument, Seq2SeqDataCollator, Collator
import numpy as np
import torch
from torch import nn, optim
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import read_config, computer_metric
from model import PW
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from transformers import TrainingArguments, Trainer
from tqdm import tqdm

# choice gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(
    level=logging.WARN,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

config_path = "./config.json"

# add local_rank and other params
parser = argparse.ArgumentParser()
parser.add_argument("--freeze", default=False, type=bool)
parser.add_argument("--local_rank", default=-1, type=int)
args = parser.parse_args()
local_rank = args.local_rank

# read config file
configs = read_config(config_path)


# Initialize the distributed training environment,Linux use nccl, windows use gloo
def init_distributed():
    torch.distributed.init_process_group(backend='gloo', init_method="env://", world_size=configs.gpus,
                                         rank=args.local_rank)


# set random seed and Add the following lines to use multiple GPUs
if torch.cuda.is_available():
    logging.warning("Cuda is available!")
    np.random.seed(configs.train.seed)
    torch.manual_seed(configs.train.seed)
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        init_distributed()
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        logging.warning(f"Find {torch.cuda.device_count()} GPUs!")
    else:
        logging.warning("Too few GPU!")
else:
    device = torch.device("cpu")
    logging.warning("Cuda is not available! Exit!")

model = PW(model_name=configs.train.model_path, freeze=args.freeze, gpu_count=torch.cuda.device_count())
# model need to gpu before DDP
model = model.to(device)
if torch.cuda.device_count() > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# get need optimizer params
need_grad_parameters = list(filter(lambda param: param.requires_grad, model.parameters()))
optimizer = optim.Adam(need_grad_parameters, lr=configs.train.lr, eps=configs.train.eps,
                       weight_decay=configs.train.weight_decay)


def train():
    model.train()
    tokenizer = AutoTokenizer.from_pretrained("./model/facebook/bart-large-cnn")
    train_dataset = MyDataSet(file_path=configs.train.file_path, tokenizen=tokenizer,
                              max_source_length=configs.train.max_document_len,
                              max_target_length=configs.train.max_summary_len)
    eval_dataset = MyDataSet(file_path=configs.eval.file_path, tokenizen=tokenizer,
                             max_source_length=configs.eval.max_document_len,
                             max_target_length=configs.eval.max_summary_len)

    #create train_dataloader and collate_fn
    collate_fn = Seq2SeqDataCollator(tokenizer, configs)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.train.batch_size,shuffle=False, collate_fn= collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=configs.eval.batch_size,shuffle=False, collate_fn= collate_fn)
    # count = 0
    # for batch in tqdm(train_dataloader):
    #     batch, result = model(batch.to(device))
    #     print(batch)
    #     print("*"*30)
    #     print(len(result))
    #     count +=1
    #     if count > 1:
    #         break
    #setting train argument
    train_args = TrainArgument()
    # print(train_args)
    compute_metrics_fn = computer_metric(tokenizer)
    # data_collator = Seq2SeqDataCollator(tokenizer, configs)
    # print(len(train_dataset))
    # print(len(eval_dataset))
    # for index , item in enumerate(train_dataset):
    #     if index < 1:
    #         print(item)
    #     else:
    #         break


    # create Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Collator,
        compute_metrics=compute_metrics_fn,
    )
    # # start train
    trainer.train()

    pass


def eval():
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("./model/facebook/bart-large-cnn")
    eval_dataset = MyDataSet(file_path=configs.eval.file_path, tokenizen=tokenizer,
                             max_source_length=configs.eval.max_document_len,
                             max_target_length=configs.eval.max_summary_len)
    # create eval_dataloader and collate_fn
    collate_fn = Seq2SeqDataCollator(tokenizer, configs)
    eval_dataloader = DataLoader(eval_dataset, batch_size=configs.eval.batch_size, shuffle=False, collate_fn=collate_fn)

    pass


if __name__ == '__main__':
    train()
