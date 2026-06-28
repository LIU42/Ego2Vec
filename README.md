# Ego2Vec: Extract Egocentric Video Embeddings

*v1.0.0 新变化：初次发布项目源代码。*

## 项目简介

本项目基于 [LAVILA](https://github.com/facebookresearch/lavila) 预训练模型，实现第一视角视频和文本描述的特征提取以及相似度计算。

## 使用说明

### 环境搭建

首先需要利用 [conda](https://anaconda.org/) 安装本项目依赖的环境。

```shell-session
conda env create -f environment.yml
```

*注：此环境配置文件可能存在冗余，可尝试自行配置环境。*

### 数据集准备

本项目的数据集格式如下。

```shell-session
datasets/
├── ek100/
│   ├── videos/
│   │   ├── P01_01_0.mp4
│   │   ├── P01_01_1.mp4
│   │   └── ...
│   ├── metadata_test.json
│   └── metadata_train.json
├── egtea/
│   └── ...
├── uestc/
│   └── ...
└── ...
```

其中每个数据集目录下 videos 子目录下存放所有视频的原始文件，部分数据集可能会以长视频的形式提供，需要自行完成分割。metadata_test.json 和 metadata_train.json 文件为所在数据集划分的标注文件，其格式如下。

```json5
{
    "labels": [
        "action1",
        "action2",
        "action3",
        ...
    ],
    "videos": [
        "video1",
        "video2",
        "video3",
        ...
    ],
    "annotations": {
        "video1": "action1",
        "video2": "action2",
        "video3": "action3",
        ...
    }
}
```

本项目已实现的三个基准数据集（[EPIC-KITCHENS-100](https://epic-kitchens.github.io/)，[EGTEA](https://cbs.ic.gatech.edu/fpv/)，[UESTC-MMEA-CL](https://ivipclab.github.io/publication_uestc-mmea-cl/mmea-cl/)）的标注文件已在代码库中提供，针对上述每个字段的解释如下。

| 字段名         | 字段描述                                  |
|:-----------:|:-------------------------------------:|
| labels      | 候选标签列表。                               |
| videos      | 视频名称列表，需要和 videos 目录下的文件名保持一致，不包含扩展名。 |
| annotations | 视频名到动作标签的映射。                          |

### 模型预训练权重准备

从 [这里](https://pan.baidu.com/share/init?surl=6O1FeuVJs5r4OO7sCKHJeA&pwd=i2xt) 下载本项目使用的模型权重文件并放入 weights 目录下。模型权重文件名包含视觉编码器名称和接受视频帧数等信息，例如

```shell-session
lavila_openai_timesformer_base.16.pth
```

表示视觉编码器为 TimeSformer-Base，接受长度为 16 帧的视频片段。

### 类别标签文本描述准备（可选）

本项目可为每个类别标签准备若干的文本描述以提升特征对齐的效果。类别标签文本描述文件为 JSON 格式，存储于 descriptions 目录下对应数据集的子目录中，其格式如下。

```json5
{
    "action1": [
        "description1",
        "description2",
        "description3",
        ...
    ],
    "action2": [
        ...
    ],
    ...
}
```

每个类别标签可提供多个描述，在特征提取的过程中会计算它们的均值。

### 特征提取及预测

准备好数据集和模型预训练权重后，使用 [Accelerate](https://hugging-face.cn/docs/accelerate/index) 运行 zeroshot.py 即可。初次运行可能需要进行一些配置，详情请参考 [官方文档](https://hugging-face.cn/docs/accelerate/index)。运行该程序需要提供的各项参数如下。

| 参数名         | 参数描述                                                    |
|:-----------:|:-------------------------------------------------------:|
| model       | 模型名称，可选 "base" (TSF-B) 和 "large" (TSF-L)。               |
| dataset     | 数据集名称，与 datasets 目录下的名称一致。                              |
| subset      | 使用的数据集划分，可选 "test" 和 "train"。                           |
| checkpoint  | 加载的权重文件名，不包含扩展名，此处读取的模型权重需要与 model 和 clip_length 参数相对应。 |
| batch_size  | 单次处理的视频样本数量。                                            |
| num_crops   | 视频帧的空间裁剪数量，可取 1 (center-crop) 和 3 (3-crops)。            |
| num_clips   | 针对一个视频样本的均匀采样片段数。                                       |
| num_workers | 数据加载器工作子进程数。                                            |
| clip_length | 输入视频编码器的视频片段长度，目前可取 4 和 16。                             |
| clip_stride | 视频帧采样间隔。                                                |
| description | 可选参数，加载的类别标签描述文件名，不包含扩展名。                               |
| prediction  | 可选参数，预测结果保存文件名，不包含扩展名，若不提供此参数则不保存预测结果。                  |
| topk        | 保存每个样本的 top-K 个类别预测结果，当 prediction 参数存在时提供。             |

*注：scripts 目录中提供已实现的三个基准数据集包装好的执行脚本。*

提取的视觉特征，类别标签特征以及类别描述特征均保存于 embeddings 目录下对应数据集的子目录下。所有的特征张量形状均为 (N, D)，其顺序和数据集划分标注文件中的列表保持一致。

同时 top-K 预测结果保存于 predictions 目录下对应数据集的子目录下。
