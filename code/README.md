# 1. Environment construction


***

* conda env 
```shell
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh
```

* python env
```shell
conda create -n recEnv python=3.8
conda activate recEnv

# conda activate py38

/data/env/miniconda3/envs/recEnv/bin/pip install -r requirements.txt
python -m pip install pandas==1.4.1
/data/env/miniconda3/envs/recEnv/bin/pip install protobuf==3.20.*
/data/env/miniconda3/envs/recEnv/bin/pip install lightgbm

```
