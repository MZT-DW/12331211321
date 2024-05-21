# XXX
This repository is the official implementation of "XXXX"
  
![image](https://github.com/MZT-DW/12331211321/blob/main/PyGP/Results/strogatz_bacres1.gif)

## Abstract
Symbolic regression is a challenging task in machine learning that aims to automatically discover highly interpretable mathematical equations from limited data. Keen efforts have been devoted to addressing this issue, yielding promising results. However, there are still bottlenecks that current methods struggle with, especially when dealing with complex problems containing various noises or with intricate underlying mathematical formulas.
In this work, we propose a novel Geometric Evolution Symbolic Regression(GESR) algorithm. Leveraging geometric semantics, the process of symbolic regression in GESR is transformed into an approximation to an unimodal target in n-dimensional topological space. Then, three key modules are proposed to enhance the approximation: (1) a new semantic gradient concept, proposed to assist the exploration, which aims to improve the accuracy of approximation; (2) a new geometric search operator, tailored for approximating the target formula directly in topological space; (3) the Levenberg-Marquardt algorithm with L2 regularization, used for the adjustment of local expression structures and the optimization of constants. With the proposal of these modules, GESR achieves state-of-the-art accuracy performance on multiple authoritative benchmark datasets and demonstrates its robustness to noise interference.

## Setup
1. Run `pip install -r requirements.txt`.
2. Install PyCuda

## Usages
You can run the codes using the following commands:
`Python ./autorun.py -save_file [path\to\save\file] -dataset_dir[path\of\dataset]`

