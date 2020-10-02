# [kaggle] M5 Forecasting - Accuracy 4th place solution

https://www.kaggle.com/c/m5-forecasting-accuracy/

This document shows how to reproduce 4th place solution for the M5 Forecasting - Accuracy competition.
If you run into any trouble with the setup/code or have any questions please contact me at monsaraida@gmail.com

# Server specs to reproduce the original solution
- AWS EC2
  - Ubuntu Server 18.04 LTS (HVM), SSD Volume Type - ami-0ac80df6eff0e70b5 (64 bit x86)
  - m5.2xlarge (8 vCPUs, 16 GB memory, 100GB storage)

# Setup environment
## Deploy code
```
cd {path-to-dir}
unzip kaggle-m5a-4th.zip
cd {path-to-dir}/kaggle-m5a-4th
```

## Install pyenv + pipenv
```
git clone https://github.com/pyenv/pyenv.git -b v1.2.16 ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
sudo apt-get update; sudo apt-get install --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y
pyenv install 3.7.6
pyenv local 3.7.6
pip install pipenv
```

## Install python requirements
```
pipenv install -r ./requirements.txt
pipenv shell
```

## Install other tools
```
sudo apt install tree unzip
```

## Extract raw data (or download raw data from kaggle)
```
cd {path-to-dir}/kaggle-m5a-4th/raw
unzip m5-forecasting-accuracy.zip
```

# How to run
## Show help
```
cd {path-to-dir}/kaggle-m5a-4th
python ./m5a_main.py --help
usage: m5a_main.py [-h] [-ddp DATA_DIR_PATH] [-opn OUTPUT_NAME]
                   [-spr SAMPLING_RATE] [-flc FOLD_ID_LIST_CSV]
                   [-plc PREDICTION_HORIZON_LIST_CSV]
```

## Optional arguments
- data_dir_path (default: '.')
  - root data directory path
- output_name (default: 'default')
  - output directory name
- sampling_rate (default: 1.0)
  - if sampling_rate is less than 1.0, items in dataset are randomly sampled
- fold_id_list_csv (default: '1941')
  - fold_id is the last day of training data
    - fold_id = 1941 : train d1 - d1941, predict d1942 - d1969 : for submission
    - fold_id = 1913 : train d1 - d1913, predict d1914 - d1941 : for validation 1
    - fold_id = 1885 : train d1 - d1885, predict d1886 - d1913 : for validation 2
    - fold_id = 1857 : train d1 - d1857, predict d1858 - d1885 : for validation 3
    - fold_id = 1829 : train d1 - d1829, predict d1830 - d1857 : for validation 4
    - fold_id = 1577 : train d1 - d1577, predict d1578 - d1605 : for validation 5
  - Multiple fold_id can be specified
    - e.g. : '1941,1913,1885,1857,1829,1577'
- prediction_horizon_list_csv (default: '7,14,21,28')
  - weekly model : '7,14,21,28'
  - monthly model : '28'
  - day-by-day model : '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28'

## Train and predict from scratch
```
cd {path-to-dir}/kaggle-m5a-4th
python ./m5a_main.py
```

## Check result
```
cd {path-to-dir}/kaggle-m5a-4th/output/default
tree .
.
├── log
│   └── m5a_main_yyyy_mmdd.log
├── model
│   └── 1941
│       ├── lgb_model_CA_1_14.bin
│       ├── lgb_model_CA_1_21.bin
│       ├── lgb_model_CA_1_28.bin
│       ├── lgb_model_CA_1_7.bin
│       ├── lgb_model_CA_2_14.bin
│       ├── lgb_model_CA_2_21.bin
│       ├── lgb_model_CA_2_28.bin
│       ├── lgb_model_CA_2_7.bin
│       ├── lgb_model_CA_3_14.bin
│       ├── lgb_model_CA_3_21.bin
│       ├── lgb_model_CA_3_28.bin
│       ├── lgb_model_CA_3_7.bin
│       ├── lgb_model_CA_4_14.bin
│       ├── lgb_model_CA_4_21.bin
│       ├── lgb_model_CA_4_28.bin
│       ├── lgb_model_CA_4_7.bin
│       ├── lgb_model_TX_1_14.bin
│       ├── lgb_model_TX_1_21.bin
│       ├── lgb_model_TX_1_28.bin
│       ├── lgb_model_TX_1_7.bin
│       ├── lgb_model_TX_2_14.bin
│       ├── lgb_model_TX_2_21.bin
│       ├── lgb_model_TX_2_28.bin
│       ├── lgb_model_TX_2_7.bin
│       ├── lgb_model_TX_3_14.bin
│       ├── lgb_model_TX_3_21.bin
│       ├── lgb_model_TX_3_28.bin
│       ├── lgb_model_TX_3_7.bin
│       ├── lgb_model_WI_1_14.bin
│       ├── lgb_model_WI_1_21.bin
│       ├── lgb_model_WI_1_28.bin
│       ├── lgb_model_WI_1_7.bin
│       ├── lgb_model_WI_2_14.bin
│       ├── lgb_model_WI_2_21.bin
│       ├── lgb_model_WI_2_28.bin
│       ├── lgb_model_WI_2_7.bin
│       ├── lgb_model_WI_3_14.bin
│       ├── lgb_model_WI_3_21.bin
│       ├── lgb_model_WI_3_28.bin
│       └── lgb_model_WI_3_7.bin
└── result
    └── 1941
        ├── feature_importance_CA_1_14.csv
        ├── feature_importance_CA_1_21.csv
        ├── feature_importance_CA_1_28.csv
        ├── feature_importance_CA_1_7.csv
        ├── feature_importance_CA_2_14.csv
        ├── feature_importance_CA_2_21.csv
        ├── feature_importance_CA_2_28.csv
        ├── feature_importance_CA_2_7.csv
        ├── feature_importance_CA_3_14.csv
        ├── feature_importance_CA_3_21.csv
        ├── feature_importance_CA_3_28.csv
        ├── feature_importance_CA_3_7.csv
        ├── feature_importance_CA_4_14.csv
        ├── feature_importance_CA_4_21.csv
        ├── feature_importance_CA_4_28.csv
        ├── feature_importance_CA_4_7.csv
        ├── feature_importance_TX_1_14.csv
        ├── feature_importance_TX_1_21.csv
        ├── feature_importance_TX_1_28.csv
        ├── feature_importance_TX_1_7.csv
        ├── feature_importance_TX_2_14.csv
        ├── feature_importance_TX_2_21.csv
        ├── feature_importance_TX_2_28.csv
        ├── feature_importance_TX_2_7.csv
        ├── feature_importance_TX_3_14.csv
        ├── feature_importance_TX_3_21.csv
        ├── feature_importance_TX_3_28.csv
        ├── feature_importance_TX_3_7.csv
        ├── feature_importance_WI_1_14.csv
        ├── feature_importance_WI_1_21.csv
        ├── feature_importance_WI_1_28.csv
        ├── feature_importance_WI_1_7.csv
        ├── feature_importance_WI_2_14.csv
        ├── feature_importance_WI_2_21.csv
        ├── feature_importance_WI_2_28.csv
        ├── feature_importance_WI_2_7.csv
        ├── feature_importance_WI_3_14.csv
        ├── feature_importance_WI_3_21.csv
        ├── feature_importance_WI_3_28.csv
        ├── feature_importance_WI_3_7.csv
        ├── feature_importance_agg_14.csv
        ├── feature_importance_agg_21.csv
        ├── feature_importance_agg_28.csv
        ├── feature_importance_agg_7.csv
        ├── feature_importance_all_14.csv
        ├── feature_importance_all_21.csv
        ├── feature_importance_all_28.csv
        ├── feature_importance_all_7.csv
        ├── holdout.csv
        ├── pred_h_14.csv
        ├── pred_h_21.csv
        ├── pred_h_28.csv
        ├── pred_h_7.csv
        ├── pred_h_all.csv
        ├── pred_v_14.csv
        ├── pred_v_21.csv
        ├── pred_v_28.csv
        ├── pred_v_7.csv
        ├── pred_v_all.csv
        └── submission.csv
```

# References
I would like to appreciate the participants and organizers of the competition very much.
I have learned a lot from this competition and community. Thank you to all the kaggle community members.

- Notebooks
  - https://www.kaggle.com/kyakovlev/m5-simple-fe
  - https://www.kaggle.com/kyakovlev/m5-lags-features
  - https://www.kaggle.com/kyakovlev/m5-custom-features
  - https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic
  - https://www.kaggle.com/dhananjay3/wrmsse-evaluator-with-extra-features

- Discussions
  - Few thoughts about M5 competition
    - https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/138881
  - Evaluation metric
    - https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834
  - Three shades of Dark
    - https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/144067
  - Moon Phase. Odd, yet helpful feature.
    - https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/154776

# Contact
- https://www.kaggle.com/monsaraida
- monsaraida@gmail.com