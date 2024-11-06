import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split

SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')
class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []; self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }

    def __len__(self):
        return len(self.labels)
    
class baselinemodel():
    def __init__(self, model_name = 'klue/bert-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)
        data = pd.read_csv('/data/ephemeral/home/code/data/train.csv')
        dataset_train, dataset_valid = train_test_split(data, test_size=0.3, random_state=SEED)
        self.f1 = evaluate.load('f1')

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.f1.compute(predictions=predictions, references=labels, average='macro')


    ### for wandb setting
    def train(self, data = None):
        if data is None:        
            data = pd.read_csv('/data/ephemeral/home/code/data/train.csv')
        dataset_train, dataset_valid = train_test_split(data, test_size=0.3, random_state=SEED)
        data_train = BERTDataset(dataset_train, self.tokenizer)
        data_valid = BERTDataset(dataset_valid, self.tokenizer)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        os.environ['WANDB_DISABLED'] = 'false'

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            do_predict=True,
            logging_strategy='steps',
            evaluation_strategy='steps',
            save_strategy='steps',
            logging_steps=100,
            eval_steps=100,
            save_steps=100,
            save_total_limit=2,
            learning_rate= 2e-05,
            adam_beta1 = 0.9,
            adam_beta2 = 0.999,
            adam_epsilon=1e-08,
            weight_decay=0.01,
            lr_scheduler_type='linear',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_f1',
            greater_is_better=True,
            seed=SEED
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=data_train,
            eval_dataset=data_valid,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
    def inference(self, dataset_test = None):
        if dataset_test is None:
            dataset_test = pd.read_csv('/data/ephemeral/home/code/data/test.csv')
        self.model.eval()
        preds = []
        logits = []

        for idx, sample in tqdm(dataset_test.iterrows(), total=len(dataset_test), desc="Evaluating"):
            inputs = self.tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logit = self.model(**inputs).logits
                logits.extend(torch.nn.Softmax(dim=1)(logit).cpu().numpy())
                pred = torch.argmax(torch.nn.Softmax(dim=1)(logit), dim=1).cpu().numpy()
                preds.extend(pred)
        dataset_test['target'] = preds
        dataset_test['logits'] = logits
        self.result = dataset_test
    def make_csv(self):
        self.result[['ID','target']].to_csv(os.path.join(BASE_DIR, 'output.csv'), index=False)