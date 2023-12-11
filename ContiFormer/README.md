# ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling
This is the code for our NeurIPS 2023 paper "[ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling](https://seqml.github.io/contiformer/)".

## Environment Dependencies

`ContiFormer` is currently part of "[PhysioPro](https://github.com/microsoft/physiopro)" project, please first clone `PhysioPro` repo and set up the required environment.

### Dependencies

`ContiFormer` requires additional dependencies:

```
pip install torchdiffeq
pip install torchcde
```

## Reproduce the experimental result for interpolating continuous-time function

Run the following command, where `cc` controls the type of spiral, `0` for the first type, `1` for the second type, `2` for mixture (see Appendix C.1.1 for more information).

* For Neural ODE, please run

```
python run.py --adjoint=1 --visualize=1 --niters=10000 --model_name Neural_ODE --noise_a=0.02 --cc=1 --train_dir ./sprial_neuralode
```

* For ContiFormer, please run

```
python run.py --adjoint=1 --visualize=1 --niters=10000 --model_name Contiformer --noise_a=0.02 --cc=1 ----train_dir ./spiral_contiformer
```

The results and visualization data will be saved to `./sprial_neuralode` and `./spiral_contiformer`.


## Reproduce the experimental result for irregular time series classification

1. Download the dataset

```bash
cd PhysioPro
mkdir data
wget http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip -P data

unzip data/Multivariate2018_ts.zip -d data/
rm data/Multivariate2018_ts.zip
```

2. Run irregular time series classification task with `ContiFormer`

```bash
# create the output directory
mkdir -p outputs/Multivariate_ts/Heartbeat
# run the train task
python -m physiopro.entry.train docs/configs/contiformer_mask_classification.yml --data.mask_ratio 0.3 --data.name Heartbeat
# tensorboard
tensorboard --logdir outputs/
```

The results will be saved to `outputs/Multivariate2018_ts/Heartbeat` directory.

### Reproducibility

#### Random seeds selection

All the experimental results are averaged over three random seed, i.e., `27, 42, 1024` (the same random seeds for event prediction task). To obtain the result for `Heartbeat` dataset with 0.3 mask ratio, please run the following three commands.

```
python -m physiopro.entry.train docs/configs/contiformer_mask_classification.yml --data.mask_ratio 0.3 --data.name Heartbeat --runtime.seed 27
python -m physiopro.entry.train docs/configs/contiformer_mask_classification.yml --data.mask_ratio 0.3 --data.name Heartbeat --runtime.seed 42
python -m physiopro.entry.train docs/configs/contiformer_mask_classification.yml --data.mask_ratio 0.3 --data.name Heartbeat --runtime.seed 1024
```

#### Hyper-parameter search

We provided hyper-parameter searching for all the compared methods. Taking ContiFormer model as an example, we search activation function in `sigmoid` and `tanh` (please refer to Appendix D.4 for more information), and report the best performance for each dataset.

To perform hyper-parameter search for the activation function, please run the following commands:

```
python -m physiopro.entry.train docs/configs/contiformer_mask_classification.yml --data.mask_ratio 0.3 --data.name Heartbeat --model.actfn_ode sigmoid
python -m physiopro.entry.train docs/configs/contiformer_mask_classification.yml --data.mask_ratio 0.3 --data.name Heartbeat --model.actfn_ode tanh
```

## Reference

You are more than welcome to cite our paper:
```
@inproceedings{chen2023contiformer,
  title={ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling},
  author={Chen, Yuqi and Ren, Kan and Wang, Yansen and Fang, Yuchen and Sun, Weiwei and Li, Dongsheng},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
