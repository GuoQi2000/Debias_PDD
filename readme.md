# Shuffle Debias
This is the github repository for Findings of EMNLP 2023 paper **"Debias NLU Datasets via Training-free Perturbations"**.

## Data
We use MNLI(debiased), FEVER(debiased) and QQP(debiased) for training, and HANS, FEVER-symmetric and PAWS for evaluation.

The `data` folder contains the data for training or evaluation. All data has been processed into JSON format.

## Code
The training code are provided by Yuanhang Tang https://github.com/yuanhangtangle/shuffle-debias.

## Train & Eval
Start with 
```
python main.py\
--base_folder exp-mnli-debiased\
--seed 21\
--data mnli_debiased\
--cuda 0
```
Replace the option `data` with `qqp` or `qqp_debiased` to evaluate on paws.
Replace the option `data` with `fever` or `fever_debiased` to evaluate on symmetric.
