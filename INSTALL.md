### Set up the python environment

```
conda create -n neuralbody python=3.7
conda activate neuralbody

# make sure that the pytorch cuda is consistent with the system cuda
# e.g., if your system cuda is 10.0, install torch 1.4 built from cuda 10.0
pip install torch==1.4.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

# install spconv
cd
git clone https://github.com/traveller59/spconv --recursive
cd spconv
git checkout abf0acf30f5526ea93e687e3f424f62d9cd8313a
export CUDA_HOME="/usr/local/cuda-10.0"
python setup.py bdist_wheel
cd dist
pip install spconv-1.2.1-cp36-cp36m-linux_x86_64.whl
```

### Set up datasets

#### ZJU-Mocap dataset

1. Download the ZJU-Mocap dataset from the official [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/ErlGNXydu1BChNQhlA4nYKwB1Yn1XEGFe56y4ipKGEoQIA?e=ET9eO1).
2. Create a soft link:
    ```
    ROOT=/path/to/neuralbody
    cd $ROOT/data
    ln -s /path/to/zju_mocap zju_mocap
    ```
