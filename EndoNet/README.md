# Pytorch-Video-Classification
Make video classification on Cholec80  using CNN with Pytorch framework.

# Environments
```bash
# 1. torch >= 1.0
conda create -n crnn
source activate crnn # or `conda activate crnn`
# GPU version
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
# CPU version
conda install pytorch-cpu torchvision-cpu -c pytorch

# 2. pip dependencies
pip install pandas scikit-learn tqdm opencv-python

# 3. prepare datasets
cd ./EndoNet_Pytorch # go to the root dir of the code
cp -r path/to/your/cholec80 ./media/ExtHDD # copy cholec80 dataset to data dir
cd ./data && python make_train_test.py # preprocess the dataset

# 4. train your network on cholec80
python train.py

# (optional)5. restore from checkpoints
python train.py -r path/to/checkpoints/file
```

To know more about the usage of scripts, run the following commands:
```bash
python train.py -h
python make_train_test.py -h
```
