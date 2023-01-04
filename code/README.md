# 1. Environment construction


***

conda env 
```shell
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh
```

python env
```shell
conda create -n recEnv python=3.8

conda activate recEnv
conda activate py38
pip install -r requirements.txt
python -m pip install pandas==1.4.1

https://blog.csdn.net/xiangfengl/article/details/126802340

mac : 
bash Miniforge3-MacOSX-arm64.sh
conda create -n recEnv python=3.6 -c ehmoussi 
```