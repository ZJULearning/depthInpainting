# LRLG： Depth Image Inpainting: Improving Low Rank Completion with Low Gradient Regularization

Table of Contents
=================
<!--ts-->
* [Introduction](#introduction)
* [Performance](#performance)
	 * [Datasets](#datasets)
	 * [Compared Algorithms](#compared-algorithms)
	 * [Results](#results)
* [Building Instruction](#building-instruction)
	 * [Prerequisites](#prerequisites)
	 * [Compile](#compile-on-ubuntudebian)
* [Usage](#usage)
* [Reference](#reference)
* [License](#license)
<!--te-->

## Introduction
LRLG is an algorithm for single depth image inpainting.

## Performance

### Datasets

+ [Depth Inpainting Dataset](http://www.cad.zju.edu.cn/home/dengcai/Data/depthinpaint/DepthInpaintData.html)

### Compared Algorithms


+ [LR] (https://epubs.siam.org/doi/abs/10.1137/080738970)
+ [LRTV] (https://ieeexplore.ieee.org/abstract/document/7113897/) 
+ LRL0: a designed based-line, which employs L0 gradient minimization.


### Results
LRLG achieved the **best** search performance among all the compared algorithms.

## Building Instruction

### Prerequisites

+ GCC 4.5+ 
+ CMake 2.8+
+ OpenCV 2.4.1+


### Compile

1. Install Dependencies:

```shell
$ sudo apt-get install g++ cmake opencv-dev
```

2. Compile:

```shell
$ git clone https://github.com/ZJULearning/depthInpainting.git
$ mkdir build/ && cd build/
$ cmake ..
$ make -j
```

## Usage
  TV norm:  ``` ./depthInpainting TV depthImage ```

  PSNR calc: ``` ./depthInpainting P depthImage mask inpainted ```

  Inpainting: ``` ./depthInpainting LRTV depthImage mask outputPath" ```

  Generating: ``` ./depthInpainting G depthImage missingRate outputMask outputMissing ```

  LowRank: ``` ./depthInpainting L depthImahe mask outputpath ```

  LRTVPHI: ``` ./depthInpainting LRTVPHI depthImage mask outputPath ```

  TVPHI norm: ``` ./depthInpainting TVPHI depthImage ```

  LRL0: ``` ./depthInpainting LRL0 depthImage mask outputPath initImage K lambda_L0 MaxIterCnt ```

  LRL0PHI: ``` ./depthInpainting LRL0PHI depthImage mask outputPath initImage K lambda_L0 MaxIterCnt ```
  
  L0: ``` /depthInpainting L0 depthImage ```

## Reference

Reference to cite when you use LRLG in a research paper:
```
@article{xue2017depth,
  title={Depth image inpainting: Improving low rank matrix completion with low gradient regularization},
  author={Xue, Hongyang and Zhang, Shengming and Cai, Deng},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={9},
  pages={4311--4320},
  year={2017},
  publisher={IEEE}
}
```

## License

LRLG is MIT-licensed.

