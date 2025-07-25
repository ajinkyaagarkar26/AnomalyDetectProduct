# -*- coding: utf-8 -*-
import platform
import argparse
import sys
sys.path.append('../')

from logdeep.models.lstm import *
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *
from logdeep.dataset.vocab import Vocab

import torch

data_dir = os.path.expanduser("./datasets")
output_dir = "./output/deeplog/"

# Config Parameters
options = dict()
options["output_dir"] = output_dir
options["train_vocab"] = output_dir + "train"
options["vocab_path"] = output_dir + "vocab.pkl"

options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Smaple
options['sample'] = "sliding_window"
options['window_size'] = 10  # if fix_window
# options['window_size'] = 20
options["min_len"] = 5

options['train_ratio'] = 1
options['valid_ratio'] = 0.1
options["test_ratio"] = 0.1

options["is_logkey"] = True
options["is_time"] = False

# Features
options['sequentials'] = options["is_logkey"]
options['quantitatives'] = False
options['semantics'] = False
options['parameters'] = options["is_time"]
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics'], options['parameters']])

# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
options["embedding_dim"] = 50
# options["vocab_size"] = 17
# options["vocab_size"] = 24
options["vocab_size"] = 24
options['num_classes'] = options["vocab_size"]


# Train
# options['batch_size'] = 128
options['batch_size'] = 32
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 200
options["n_epochs_stop"] = 10
options['lr_step'] = (options['max_epoch'] - 20, options['max_epoch'])
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "deeplog"
options['save_dir'] = options["output_dir"] + "deeplog/"

# Predict
options['model_path'] = options["save_dir"] + "bestloss.pth"
options['num_candidates'] = 9
options["threshold"] = None
options["gaussian_mean"] = 0
options["gaussian_std"] = 0

print("Features logkey:{} time: {}".format(options["is_logkey"], options["is_time"]))
print("Device:", options['device'])

seed_everything(seed=1234)


Model = Deeplog(input_size=options['input_size'],
                hidden_size=options['hidden_size'],
                num_layers=options['num_layers'],
                vocab_size=options["vocab_size"],
                embedding_dim=options["embedding_dim"])

def train():
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    print("options[sequentials]...", options['sequentials'])
    predicter = Predicter(Model, options)
    os = predicter.predict_unsupervised()
    # os = predicter.predict_unsupervised_2()
    return os

def vocab():
    with open(options["train_vocab"], 'r') as f:
        logs = f.readlines()
    vocab = Vocab(logs)
    print("vocab_size", len(vocab))
    vocab.save_vocab(options["vocab_path"])
    return(len(vocab))

    

def process_vocab(options):
    with open(options["train_vocab"], 'r') as f:
        logs = f.readlines()
    vocab = Vocab(logs)
    print("vocab_size", len(vocab))
    vocab.save_vocab(options["vocab_path"])
    options["vocab_size"] = len(vocab)
    options['num_classes'] = options["vocab_size"]
    print("Vocab saved at", options["vocab_path"])
    print("saved vocab_size", options["vocab_size"])
    return len(vocab)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')
    predict_parser.add_argument('--mean', type=float, default=0, help='error gaussian distribution mean')
    predict_parser.add_argument('--std', type=float, default=0, help='error gaussian distribution std')
    predict_parser.add_argument('--vocab_size', type=int, help='vocabulary size for the model')

    vocab_parser = subparsers.add_parser('vocab')
    vocab_parser.set_defaults(mode='vocab')

    args = parser.parse_args()
    print("arguments", args)

    if args.mode == 'train':
        process_vocab(options)
        train()

    elif args.mode == 'predict':
        anomalyCS = predict()
        print("anomalyCS", anomalyCS)

    elif args.mode == 'vocab':
        vocab_size = process_vocab(options)
        print(f"Vocab size: {vocab_size}")
        sys.exit(vocab_size)