# Graph Matching with Noisy Correspondence

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-matching-with-bi-level-noisy/graph-matching-on-pascal-voc)](https://paperswithcode.com/sota/graph-matching-on-pascal-voc?p=graph-matching-with-bi-level-noisy)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-matching-with-bi-level-noisy/graph-matching-on-spair-71k)](https://paperswithcode.com/sota/graph-matching-on-spair-71k?p=graph-matching-with-bi-level-noisy)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-matching-with-bi-level-noisy/graph-matching-on-willow-object-class)](https://paperswithcode.com/sota/graph-matching-on-willow-object-class?p=graph-matching-with-bi-level-noisy)


This repo contains the code and data of our ICCV'2023 paper. Our work has also been included by famous graph matching open-source projects [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch) 
[![GitHub stars](https://img.shields.io/github/stars/Thinklab-SJTU/ThinkMatch.svg?style=social&label=Star&maxAge=8640)](https://GitHub.com/Thinklab-SJTU/ThinkMatch/). 
> Yijie Lin, Mouxing Yang, Jun Yu, Peng Hu, Changqing Zhang, Xi Peng. Graph Matching with Bi-level Noisy Correspondence. ICCV, 2023.  [[paper]](https://arxiv.org/pdf/2212.04085.pdf) 





## Background of Graph Matching
Graph Matching (GM) is a fundamental yet challenging problem in computer vision, pattern recognition and data mining. GM aims to find node-to-node correspondence among multiple graphs, by solving an NP-hard combinatorial problem named Quadratic Assignment Problem (QAP).

Graph matching techniques have been applied to the following applications:

* [Image correspondence](https://arxiv.org/pdf/1911.11763.pdf)
  
  <img src="https://thinkmatch.readthedocs.io/en/latest/_images/superglue.png" alt="Superglue, CVPR 2020" width="45%">

* [Molecules matching](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Combinatorial_Learning_of_Graph_Edit_Distance_via_Dynamic_Embedding_CVPR_2021_paper.pdf)

  <img src="https://thinkmatch.readthedocs.io/en/latest/_images/molecules.png" alt="Molecules matching, CVPR 2021" width="50%">

* and more...

## Introduction to Noisy Correspondence

In this paper, we introduce a novel and widely existing problem in graph matching (GM) and focus on the scenario of visual image keypoint matching. 
As shown below, the inaccurate annotations will inevitably lead to **Bi-level Noisy Correspondence (BNC)** problem. 
Due to the poor recognizability and viewpoint differences between images, it is inevitable to inaccurately annotate some keypoints with offset and confusion, leading to the mismatch between two associated nodes (NNC). The noisy node-to-node correspondence will further contaminate the edge-to-edge correspondence (ENC).

<img src="https://github.com/Lin-Yijie/Graph-Matching-Networks/blob/main/COMMON/docs/images/nc_example.png" alt="COMMON, ICCV 2023" width="80%">




## Get Started

### Docker (RECOMMENDED)

Some of the module needs C++ supporting and we highly encouraged to directly use the docker environment. Get the recommended docker image by
```bash
docker pull runzhongwang/thinkmatch:torch1.10.0-cuda11.3-cudnn8-pyg2.0.3-pygmtools0.3.8
docker run --gpus all --name thinkmatch -p 10000:22 -it runzhongwang/thinkmatch:torch1.10.0-cuda11.3-cudnn8-pyg2.0.3-pygmtools0.3.8
pip install ortools==9.4.1874
```

Note we train our model on a single 3090 GPU. The training time is about 9 hours for Pascal VOC and 4 hours for Spair71k.


### Manual configuration (for Ubuntu, NOT RECOMMENDED)
The below python environment is provided by [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch) and we do not guarantee the integrity.

1. Install and configure Pytorch 1.6 (with GPU support). 
1. Install ninja-build: ``apt-get install ninja-build``
1. Install python packages: 
    ```bash
    pip install tensorboardX scipy easydict pyyaml xlrd xlwt pynvml pygmtools
   ```
1. Install building tools for LPMP: 
    ```bash
    apt-get install -y findutils libhdf5-serial-dev git wget libssl-dev
    
    wget https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1.tar.gz && tar zxvf cmake-3.19.1.tar.gz
    cd cmake-3.19.1 && ./bootstrap && make && make install
    ```

1. Install and build LPMP:
    ```bash
   python -m pip install git+https://git@github.com/rogerwwww/lpmp.git
   ```
   You may need ``gcc-9`` to successfully build LPMP. Here we provide an example installing and configuring ``gcc-9``: 
   ```bash
   apt-get update
   apt-get install -y software-properties-common
   add-apt-repository ppa:ubuntu-toolchain-r/test
   
   apt-get install -y gcc-9 g++-9
   update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
   ```

1. Install torch-geometric:
    ```bash
    export CUDA=cu101
    export TORCH=1.6.0
    /opt/conda/bin/pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    /opt/conda/bin/pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    /opt/conda/bin/pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    /opt/conda/bin/pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    /opt/conda/bin/pip install torch-geometric==1.6.3
   ```

1. If you have configured ``gcc-9`` to build LPMP, be sure to switch back to ``gcc-7`` because this code repository is based on ``gcc-7``. Here is also an example:

    ```bash
    update-alternatives --remove gcc /usr/bin/gcc-9
   update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
   ```

### Available datasets

Note: All following datasets can be automatically downloaded and unzipped by `pygmtools` in this code, but we recommend downloading the dataset yourself as it is much faster.

1. PascalVOC-Keypoint

    1. Download [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) and make sure it looks like ``data/PascalVOC/TrainVal/VOCdevkit/VOC2011``
    
    1. Download keypoint annotation for VOC2011 from [Berkeley server](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) or [google drive](https://drive.google.com/open?id=1D5o8rmnY1-DaDrgAXSygnflX5c-JyUWR) and make sure it looks like ``data/PascalVOC/annotations``
    
    1. The train/test split is available in ``data/PascalVOC/voc2011_pairs.npz``. **This file must be added manually.**

    Please cite the following papers if you use PascalVOC-Keypoint dataset:
    ```
    @article{EveringhamIJCV10,
      title={The pascal visual object classes (voc) challenge},
      author={Everingham, Mark and Van Gool, Luc and Williams, Christopher KI and Winn, John and Zisserman, Andrew},
      journal={International Journal of Computer Vision},
      volume={88},
      pages={303–338},
      year={2010}
    }
    
    @inproceedings{BourdevICCV09,
      title={Poselets: Body part detectors trained using 3d human pose annotations},
      author={Bourdev, L. and Malik, J.},
      booktitle={International Conference on Computer Vision},
      pages={1365--1372},
      year={2009},
      organization={IEEE}
    }
    ```
1. Willow-Object-Class
   
    1. Download [Willow-ObjectClass dataset](http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip)
    
    1. Unzip the dataset and make sure it looks like ``data/WillowObject/WILLOW-ObjectClass``

    Please cite the following paper if you use Willow-Object-Class dataset:
    ```
    @inproceedings{ChoICCV13,
      author={Cho, Minsu and Alahari, Karteek and Ponce, Jean},
      title = {Learning Graphs to Match},
      booktitle = {International Conference on Computer Vision},
      pages={25--32},
      year={2013}
    }
    ```

1. SPair-71k

    1. Download [SPair-71k dataset](http://cvlab.postech.ac.kr/research/SPair-71k/)

    1. Unzip the dataset and make sure it looks like ``data/SPair-71k``

    Please cite the following papers if you use SPair-71k dataset:

    ```
    @article{min2019spair,
       title={SPair-71k: A Large-scale Benchmark for Semantic Correspondence},
       author={Juhong Min and Jongmin Lee and Jean Ponce and Minsu Cho},
       journal={arXiv prepreint arXiv:1908.10543},
       year={2019}
    }
    
    @InProceedings{min2019hyperpixel, 
       title={Hyperpixel Flow: Semantic Correspondence with Multi-layer Neural Features},
       author={Juhong Min and Jongmin Lee and Jean Ponce and Minsu Cho},
       booktitle={ICCV},
       year={2019}
    }
    ```
For more information, please see [pygmtools](https://pypi.org/project/pygmtools/).

## Run the Experiment


Run training and evaluation
```bash
python train_eval.py --cfg path/to/your/yaml
```

and replace ``path/to/your/yaml`` by path to your configuration file, e.g.
```bash
python train_eval.py --cfg experiments/vgg16_common_willow.yaml
```

Default configuration files are stored in``experiments/`` and you are welcomed to try your own configurations.

### File Organization

```
├── experiments
│   the hyperparameter setting of experiments
├── models
│     └── COMMON
│         the module and training pipeline of COMMON
│          ├── model.py
│          │   the implementation of training/evaluation procedures of COMMON
│          ├── model_config.py
│          │   the declaration of model hyperparameters
│          └── sconv_archs.py
│              the implementation of spline convolution (SpilneCNN) operations, the same with BBGM
├── src
│  the source code of the Graph Matching, from ThinkMatch
│      └── loss_func.py
│          the implementation of loss functions 
├── eval.py
|   evlaution script
└── train_eval.py
    training script
```




## Pretrained Models
We provides pretrained models. The model weights are available via [google drive](https://drive.google.com/drive/folders/1OdpCanr_aO5GxfC3gUXFqWXk8cZBM-nU?usp=share_link)

To use the pretrained models, firstly download the weight files, then add the following line to your yaml file:
```yaml
PRETRAINED_PATH: path/to/your/pretrained/weights
```



## Results

For benchmark results, please refer to https://paperswithcode.com/task/graph-matching and [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch).

### PascalVOC

| model                                                        | year | aero   | bike   | bird   | boat   | bottle | bus    | car    | cat    | chair  | cow    | table  | dog    | horse  | mbkie  | person | plant  | sheep  | sofa   | train  | tv     | mean   |
| ------------------------------------------------------------ | ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [COMMON](https://arxiv.org/pdf/2212.04085.pdf) | 2023 | 0.6560 | 0.7520 | 0.8080 | 0.7950    |0.8930 | 0.9230 | 0.9010 | 0.8180 | 0.6160 | 0.8070| 0.9500 | 0.8200 |    0.8160    | 0.7950 | 0.6660 |    0.9890 | 0.7890 | 0.8090 | 0.9930 |    0.9380 | 0.8270 |  

### Willow Object Class

| model                                                        | year | remark          | Car    | Duck   | Face   | Motorbike | Winebottle | mean   |
| ------------------------------------------------------------ | ---- | --------------- | ------ | ------ | ------ | --------- | ---------- | ------ |
| [COMMON](https://arxiv.org/pdf/2212.04085.pdf) | 2023 | -             | 0.9760 | 0.9820 | 1.0000 | 1.0000 | 0.9960     | 0.9910 |

### SPair-71k

| model                                                        | year | aero   | bike   | bird   | boat   | bottle | bus    | car    | cat    | chair  | cow    | dog    | horse  | mtbike | person | plant  | sheep  | train  | tv     | mean   |
| ------------------------------------------------------------ | ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [COMMON](https://arxiv.org/pdf/2212.04085.pdf)    | 2023 | 0.7730 | 0.6820 | 0.9200 | 0.7950 | 0.7040 | 0.9750 | 0.9160 | 0.8250 | 0.7220 | 0.8800 | 0.8000| 0.7410 | 0.8340 | 0.8280 | 0.9990 | 0.8440 | 0.9820 | 0.9980| 0.8450 |

## Credits and Citation
Please cite the following paper if you use this model in your research:
```
@article{lin2023graph,
  title={Graph Matching with Bi-level Noisy Correspondence},
  author={Lin, Yijie and Yang, Mouxing and Yu, Jun and Hu, Peng and Zhang, Changqing and Peng, Xi},
  journal={IEEE International Conference on Computer Vision},
  year={2023}
}
```

## Acknowledgement
This repo is built upon the framework of [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch) and the network structure of [BBGM](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730409.pdf), thanks for their excellent work!

