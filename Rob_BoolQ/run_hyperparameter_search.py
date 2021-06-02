"""Run a hyperparameter search on a RoBERTa model fine-tuned on BoolQ.

Example usage:
    python run_hyperparameter_search.py BoolQ/
"""
import argparse
import boolq
import data_utils
import finetuning_utils
import json
import pandas as pd

from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a RoBERTa model on the BoolQ dataset."
)
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing the BoolQ dataset. Can be downloaded from https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip.",
)

args = parser.parse_args()

# Since the labels for the test set have not been released, we will use half of the
# validation dataset as our test dataset for the purposes of this assignment.
train_df = pd.read_json(f"{args.data_dir}/train.jsonl", lines=True, orient="records")
val_df, test_df = train_test_split(
    pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
    test_size=0.5,
)
from transformers import RobertaTokenizerFast, Trainer, TrainingArguments
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch



tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
train_data = boolq.BoolQDataset(train_df, tokenizer)
val_data = boolq.BoolQDataset(val_df, tokenizer)
test_data = boolq.BoolQDataset(test_df, tokenizer)

## TODO: Initialize a transformers.TrainingArguments object here for use in
## training and tuning the model. Consult the assignment handout for some
## sample hyperparameter values.


training_args = TrainingArguments(
    output_dir="/scratch/ks4765/",
    do_train=True,
    do_eval=True,
    per_gpu_train_batch_size=8,
    num_train_epochs=3, 
    evaluation_strategy = "epoch", # evaluate at the end of every epoch
)

## TODO: Initialize a transformers.Trainer object and run a Bayesian
## hyperparameter search for at least 5 trials (but not too many) on the 
## learning rate. Hint: use the model_init() and
## compute_metrics() methods from finetuning_utils.py as arguments to
## Trainer(). Use the hp_space parameter in hyperparameter_search() to specify
## your hyperparameter search space. (Note that this parameter takes a function
## as its value.)
## Also print out the run ID, objective value,
## and hyperparameters of your best run.


model = finetuning_utils.model_init
compute_metrics = finetuning_utils.compute_metrics

trainer = Trainer(
    args=training_args,
    model_init=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

hp_space = lambda _ : {
     'learning_rate': tune.uniform(1e-5,5e-5)
     }
objective = lambda _some_dict : _some_dict['eval_loss']

optimization = trainer.hyperparameter_search(hp_space,n_trials=5,compute_objective=objective,
       backend="ray",search_alg=BayesOptSearch(),mode='min',log_to_file=True)

print("===========Summary============")
print("Best run: ", optimization.run_id)
print("Best learning rate: ", optimization.hyperparameters)
print("Best objective val: ", optimization.objective)


'''
# EXTRA CREDIT WORKS: 

# Please uncomment the whole block to run

# Extra credit attempts using different search algorithm:

# Grid Search
from ray.tune.suggest.basic_variant import BasicVariantGenerator 
hp_space_grid = lambda _ : {
  'learning_rate' : tune.grid_search([0.000015,0.000025,0.000035,0.000045,0.000055])
       }

grid_alg = BasicVariantGenerator()
grid_res= trainer.hyperparameter_search(hp_space_grid,n_trials=1,backend='ray',compute_objective=objective,mode='min',
       search_alg=grid_alg,log_to_file=True)

print("===========Summary============")
print("Best run: ", grid_res.run_id)
print("Best learning rate: ", grid_res.hyperparameters)
print("Best objective val: ", grid_res.objective)



# Random Search
from ray.tune.suggest.basic_variant import BasicVariantGenerator

hp_space_rand = lambda _ : {
   'learning_rate' : tune.uniform(1e-5,5e-5)
        }

rand_alg = BasicVariantGenerator()
rand_res= trainer.hyperparameter_search(hp_space_rand,n_trials=5,backend='ray',compute_objective=objective,mode='min',
        search_alg=rand_alg,log_to_file=True)

print("===========Summary============")
print("Best run: ", rand_res.run_id)
print("Best learning rate: ", rand_res.hyperparameters)
print("Best objective val: ", rand_res.objective)


# HyperOptSearch Search
# Note: for this one, I choose to use "maximizing the computing_metrics (f1, precision, recall)" to search for parameters
# HOWEVER FAILED TO RUN!! This is mentioend in my report! 
from ray.tune.suggest.hyperopt import HyperOptSearch
import hyperopt as hp

hp_space_hyperOptSearch = lambda _ :{
    'learning_rate': hp.uniform('learning_rate',1e-5,5e-5)
    }

hyperOpt = trainer.hyperparameter_search(hp_space_hyperOptSearch,n_trials=5,mode="max",backend="ray",
        search_alg=HyperOptSearch(),log_to_file=True)

print("===========Summary============")
print("Best run: ", hyperOpt.run_id)
print("Best learning rate: ", hyperOpt.hyperparameters)
print("Best objective val: ", hyperOpt.objective)


'''

