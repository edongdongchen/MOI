# Multi-Operator Imaging (MOI) in PyTorch

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]([https://arxiv.org/abs/2201.12151](https://arxiv.org/abs/2201.12151))
[![GitHub Stars](https://img.shields.io/github/stars/edongdongchen/MOI?style=social)](https://github.com/edongdongchen/MOI)

[Unsupervised Learning From Incomplete Measurements for Inverse Problems](https://arxiv.org/pdf/2201.12151.pdf)

[Julián Tachella](https://tachella.github.io/), [Dongdong Chen](https://dongdongchen.com), [Mike E. Davies](https://www.research.ed.ac.uk/en/persons/michael-davies).

CNRS, France; The University of Edinburgh, UK

In NeurIPS 2022



## Run the code

1. Requirements: configure the environment by following [environment.yml](https://github.com/edongdongchen/MOI/blob/main/environment.yml)

2. find the implementation of 'Multi-Operator Imaging (MOI)' at [moi.py](https://github.com/edongdongchen/MOI/blob/main/moi/moi.py)

3. download datasets from the below source, then preprocess (see our paper for details)
   and move the datasets under the folders: `../dataset/mri`, `../dataset/CelebA`, and `../dataset/mnist`, repectively:
   * mnist: built-in dataset in PyTorch
   * CelebA: https://www.kaggle.com/jessicali9530/celeba-dataset
   * fastMRI (only the subset 'Knee MRI'): https://fastmri.med.nyu.edu/

4. **Train**: run the below scripts to train/test the models:
   * run [demo_train.py](https://github.com/edongdongchen/MOI/blob/main/demo_train.py) to train MOI for CS-MNIST, Inpainting-MNIST, Inpainting-CelebA, and MRI-fastMRI tasks, respectively.
       All the trained models can be found in the folder './ckp/'
   * or run [train_bash.py](https://github.com/edongdongchen/MOI/blob/main/train_bash.py) to train MOI models on all tasks.
   ```
   bash train_bash.sh
   ```

5. **Test**: run [demo_test.py](https://github.com/edongdongchen/MOI/blob/main/demo_test.py) to test the performance (PSNR) of a trained model on a specific task.
   ```
   python3 demo_test.py
   ``` 


## Citation
If you use this code for your research, please cite our papers.
  ```
  @article{tachella2022sampling,
  title={Unsupervised Learning From Incomplete Measurements for Inverse Problems},
  author={Tachella, Juli{\'a}n and Chen, Dongdong and Davies, Mike},
  journal={arXiv preprint arXiv:2201.12151},
  year={2022}}
  ```
