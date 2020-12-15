import torch
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from torch.utils.data.dataset import Dataset
from typing import List, Optional, Union
from enum import Enum
from filelock import FileLock
import time
import h5py
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from torch.nn import BCEWithLogitsLoss
from tqdm.auto import tqdm
import timeit
from transformers import LongformerForSequenceClassification
from transformers import DataProcessor
from transformers.data.processors.glue import InputExample, InputFeatures
from transformers import HfArgumentParser
from transformers import Trainer
from transformers import TrainingArguments
from transformers import glue_compute_metrics
from transformers import glue_output_modes
from transformers import glue_tasks_num_labels
from transformers import set_seed
from transformers import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from transformers import WEIGHTS_NAME
from transformers import AdamW
from transformers import AutoConfig
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import squad_convert_examples_to_features
from transformers import PreTrainedTokenizer
from transformers.data.metrics.squad_metrics import compute_predictions_log_probs
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.metrics.squad_metrics import squad_evaluate

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

class Split(Enum):
  train = "train"
  dev = "dev"
  test = "test"

@dataclass
class ModelArguments:
  """ Arguments pertaining to which model/config/tokenizer we are going to fine-tune from. """
  model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
  tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
  cache_dir: Optional[str] = field(default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})

@dataclass
class DataTrainingArguments:
  """ Arguments pertaining to what data we are going to input our model for training and eval. """
  data_dir: Optional[str] = field(default='', metadata={"help": "Path for cached train dataset"})
  max_seq_length: int = field(default=512, metadata={"help": "Max input length for the source text"})
  limit_length : Optional[int] = field(default=None, metadata={"help": "Max no of samples"})
  task_name : Optional[str] = field(default="multi-label", metadata={"help": "task name"})
  overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets."})

class DnaProcessor(DataProcessor):
  """Processor for the SST-2 data set (GLUE version)."""
  train: List = []
  test: List = []
  all_labels = []
  def __init__(self):
    train_h5 = h5py.File('../data_seq/train.hdf5')
    X_train = train_h5['X_train'][:]
    y_train = torch.tensor(train_h5['y_train'][:])
    train_h5.close()
    self.train = {"text" : X_train, "labels" : y_train}
    valid = np.load('../data_seq/valid.npz')
    X_valid = valid['arr_0'][:]
    y_valid = torch.tensor(valid['arr_1'][:]) 
    self.test = {"text" : X_valid, "labels" : y_valid}
    self.all_labels = np.array(np.array(range(0, 919)))
  def get_train_examples(self):
    return self._create_examples("train")
  def get_dev_examples(self):
    return self._create_examples("dev")
  def get_labels(self):
    return self.all_labels
  def _create_examples(self, set_type):
    if set_type == "train":
      lines = self.train
    elif set_type == "dev":
      lines = self.test
    examples = []
    for i in range(0, lines['text'].shape[0]):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = lines["text"][i]
      label = lines["labels"][i]
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

@dataclass(frozen=True)
class MultiInputExample:
  input_ids: List
  attention_mask: List
  label_ids: List

def convert_examples_to_multi_features(examples, tokenizer, max_length=512, task=None, label_list=None, 
  output_mode=None, pad_on_left=False, pad_token=0, pad_token_segment_id=0, mask_padding_with_zero=True):
  if label_list is None:
    label_list = processor.get_labels()
  features = []
  for (ex_index, example) in enumerate(examples):
    print("Processing:", ex_index)
    inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, pad_to_max_length=True)
    input_ids , attention_mask = inputs["input_ids"] , inputs["attention_mask"]
    labels = example.label
    feature = {"input_ids": input_ids, "attention_mask": attention_mask, "label_ids": labels}
    features.append(MultiInputExample(**feature))
  return features

class CustomDataset(Dataset):
  """ This will be superseded by a framework-agnostic approach soon. """
  args: DataTrainingArguments
  output_mode: str
  features: List[InputFeatures]
  examples: List[InputExample]
  dataset: TensorDataset
  def __init__(self, args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, limit_length: Optional[int] = None, mode: Union[str, Split] = Split.train):
    self.args = args
    self.processor = processor
    self.output_mode = "classification"
    if isinstance(mode, str):
      mode = Split[mode]
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name, limit_length))
    lock_path = cached_features_file + ".lock"
    with FileLock(lock_path):
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
          start = time.time()
          obj = torch.load(cached_features_file)
          self.features = obj["features"]
          self.examples = obj["examples"]
        else:
          if mode == Split.dev:
            examples = self.processor.get_dev_examples()
          elif mode == Split.test:
            examples = self.processor.get_test_examples()
          else:
            examples = self.processor.get_train_examples()
          self.examples = examples
          self.features = convert_examples_to_multi_features(examples=examples,tokenizer=tokenizer,max_length=args.max_seq_length)
          start = time.time()
          torch.save({"examples" : self.examples,"features" : self.features,}, cached_features_file)
  def __len__(self):
    return len(self.features)
  def __getitem__(self, i) -> InputFeatures:
    return self.features[i]
  def get_features(self):
    return self.features
  def get_examples(self):
    return self.examples

def multi_data_collator(features) -> Dict[str, torch.Tensor]:
  first = features[0]
  if hasattr(first, "label_ids") and first.label_ids is not None:
    labels = torch.stack([f.label_ids for f in features])
    batch = {"labels": labels}
  else:
    batch = {}
  for k, v in vars(first).items():
    if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
      batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
  return batch


config = {
  "model_name" : 'allenai/longformer-base-4096',
  "model_name_or_path": 'allenai/longformer-base-4096',
  "max_seq_length": 1024 ,
  "output_dir": './models',
  "overwrite_output_dir": True,
  "per_gpu_train_batch_size": 4,
  "per_gpu_eval_batch_size": 4,
  "overwrite_cache": True,
  "learning_rate": 1e-4,
  "num_train_epochs": 3,
  "do_train": True,
  "do_eval" : True,
  "save_steps": 5000,
  "save_total_limit": 2,
}

import json
with open('args.json', 'w') as f:
  json.dump(config, f)

class_weights = torch.load(r"/data0/abdul/DeepLearningProject/data_seq/class_weights")
class LongformerForMultiLabelSequenceClassification(LongformerForSequenceClassification):
  def __init__(self, config):
    super().__init__(config)
  def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, token_type_ids=None,
    position_ids=None, inputs_embeds=None, labels=None, output_attentions=None,):
    if global_attention_mask is None:
      global_attention_mask = torch.zeros_like(input_ids)
      global_attention_mask[:, 0] = 1
    outputs = self.longformer(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions)
    sequence_output = outputs[0]
    logits = self.classifier(sequence_output)
    outputs = (logits,) + outputs[2:]
    if labels is not None:
      loss_fct = BCEWithLogitsLoss(pos_weight=class_weights)
      loss = loss_fct(logits.type(torch.DoubleTensor), labels.type(torch.DoubleTensor))
      outputs = (loss,) + outputs
    return outputs  # (loss), logits, (hidden_states), (attentions)

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_json_file(json_file="args.json")

processor = DnaProcessor()
set_seed(training_args.seed)
label_list = processor.get_labels()
config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir if model_args.cache_dir else None, num_labels=len(label_list))
tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path, cache_dir=model_args.cache_dir if model_args.cache_dir else None, use_fast=True)

train_dataset = CustomDataset(data_args, tokenizer=tokenizer, limit_length=1024)
eval_dataset = CustomDataset(data_args, tokenizer=tokenizer, mode="dev", limit_length=1024)

model = LongformerForMultiLabelSequenceClassification.from_pretrained(model_args.model_name_or_path, from_tf=bool(".ckpt" in model_args.model_name_or_path), config=config, cache_dir=model_args.cache_dir if model_args.cache_dir else None)
trainer = Trainer(model=model, args=training_args, data_collator=multi_data_collator, train_dataset=train_dataset, eval_dataset=eval_dataset)

trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
trainer.save_model()
