INSTALL
====
Tested with PyTorch 1.6 + CUDA 10.2

Step 1: Install apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Step 2: Install this repo
```bash
nvcc --version   # Should be 10.2
cd pytorch_minimax
python setup.py install
cd ..
cd quantize
pip install -e .
cd ..
```

Memory Saving Training
====

Quick test
```bash
cd resnet
wget https://github.com/cjf00000/RN50v1.5/releases/download/v0.1/results.tar.gz
tar xzvf results.tar.gz
python main.py <data config> <quantize config> --workspace results/tmp --evaluate --training-only --resume results/exact_seed0/checkpoint-10.pth.tar --resume2 results/exact_seed0/checkpoint-10.pth.tar  ~/data/cifar100 
```

Full training
```
mkdir results/ca2
python main.py --dataset cifar10 --gather-checkpoints --arch preact_resnet56 --gather-checkpoints --workspace results/ca2 --batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 --weight-decay 1e-4 --epochs 200 --ca=True --cabits=2 -c quantize -j 0  ~/data/cifar10
```

Results
----

CIFAR10

```--dataset cifar10 --arch preact_resnet56 --epochs 200 --num-classes 10 -j 0 --weight-decay 1e-4 --batch-size 128 --label-smoothing 0```

| *quantize config* | *Overall Var* |
|--------|----------|
| -c qlinear --ca=True --cabits=4 --ibits=4 --pergroup=False --perlayer=False | 0.009703520685434341 |
| -c qlinear --ca=True --cabits=2 --ibits=2 --pergroup=False --perlayer=False | 0.21035686135292053 |
| -c qlinear --ca=True --cabits=2 --ibits=2 --pergroup=True --perlayer=False | 0.0181496012955904 |
| -c qlinear --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False | 0.002921732608228922 |
| -c qlinear --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True | 0.001438044011592865 |
| -c quantize --ca=True --cabits=4 --ibits=4 --pergroup=False --perlayer=False | 0.011829948052763939 |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=False --perlayer=False | 0.29687657952308655 |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=True --perlayer=False | 0.023775238543748856 |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False | 0.0034970948472619057 |  
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True | 0.0017123270081356168 | 

CIFAR100

```--dataset cifar10 --arch preact_resnet56 --epochs 200 --num-classes 100 -j 0 --weight-decay 1e-4 --batch-size 128 --label-smoothing 0```

| *quantize config* | *Overall Var* |
|--------|----------|
| -c qlinear --ca=True --cabits=4 --ibits=4 --pergroup=False --perlayer=False | 0.03486475348472595 |
| -c qlinear --ca=True --cabits=2 --ibits=2 --pergroup=False --perlayer=False | 0.7869864702224731 |
| -c qlinear --ca=True --cabits=2 --ibits=2 --pergroup=True --perlayer=False | 0.09328116476535797 |
| -c qlinear --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False | 0.04863186180591583 |
| -c qlinear --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True | 0.018564047291874886 |
| -c quantize --ca=True --cabits=4 --ibits=4 --pergroup=False --perlayer=False | 0.03873932361602783 |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=False --perlayer=False | 0.9092444181442261 |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=True --perlayer=False | 0.1020037978887558 |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False | 0.05299000069499016 |  
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True | 0.0214 | 