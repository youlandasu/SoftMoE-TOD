# Many Hands Make Light Work: Task-Oriented Dialogue System with Module-Based Mixture-of-Experts


# Prepare Environment

## Create Conda venv
```bash
conda create -n toatod python=3.8
conda activate toatod
```

## Install requirements
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install wandb
wandb login
wandb init
```
# Prepare Data

## MultiWOZ 2.1
```bash
cd data/multiwoz21
bash data_preparation.sh
```
## MultiWOZ 2.2
- MultiWOZ 2.2 Data Preprocessing
Download the dataset from [MultiWOZ 2.2](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2). Then using the following script to process.
```bash
cd data/multiwoz22
bash data_preparation.sh
```
## Banking77, CLINC150, HWU64
```bash
cd data/banking77
bash banking77_preparation.sh

cd ../clinc150
bash clinc150_preparation.sh

cd ../hwu64
bash ../hwu64_preparation.sh
```

## Download [PPTOD](https://github.com/awslabs/pptod/tree/main) small and base checkpoints
```bash
cd checkpoints

cd checkpoints
chmod +x ./download_pptod_small.sh
./download_pptod_small.sh

cd checkpoints
chmod +x ./download_pptod_base.sh
./download_pptod_base.sh
```

# Experiments
## Start Training NLU
```bash
cd IC
chmod +x train_run_mmoe.sh
bash train_run_mmoe.sh (bash)
./train_run_mmoe.sh (zsh)
```

## Evaluate NLU
```bash
cd IC
chmod +x test_run.sh
bash test_run.sh (bash)
./test_run.sh (zsh)
```

## Start Training NLG
```bash
cd E2E_TOD
chmod +x train_run_mmoe.sh
bash train_run_mmoe.sh (bash)
./train_run_mmoe.sh (zsh)
```

## Evaluate NLG
```bash
cd E2E_TOD
chmod +x test_run.sh
bash test_run.sh (bash)
./test_run.sh (zsh)
```