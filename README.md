# SPECTRAL-SETR

这个仓库用于做分割任务，模型论文：[SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](http://arxiv.org/abs/2105.15203)

## 1. 环境配置

### 创建环境

```shell
# create environment
conda create -n torch python=3.7 -y 

# Install pytorch
conda install pytorch=1.8.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge -y
```
### 其他依赖

```
opencv-python
tqdm
pillow
pandas
timm
matplotlib
```

## 2. 数据集准备

先根据 `utils/gaofen_prepare.py` 中的注释，修改自己的本机路径，接着执行 `python utils/gaofen_prepare.py`。

执行完毕后，应该可以看到文件结构：

```
"""
├─image
├─label
├─val.csv
└─train.csv
"""
```

## 3. quick start

准备好数据集后，在 `configs/segformer_b4_gaofen.py`中，修改`dataset_path`，然后 `python train.py -config configs/segformer_b4_gaofen.py`。

## 3. 配置文件

配置文件都在 `configs/*` 目录下，配置文件命名规则 `{model_name}_{dataset}.py`，在配置文件中，定义使用的Model、dataset、hyperparameters等信息。

## 4. train

`python train.py -config configs/{model_name}_{dataset}.py`

## 5. eval

`python test.py -config configs/{model_name}_{dataset}.py -weight work/models/{model}`


