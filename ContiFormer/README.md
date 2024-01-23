# ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling

This is the code for our NeurIPS 2023 paper "[ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling](https://seqml.github.io/contiformer/)".

## Updates

- [2023/12/11] Support continuous-time function approximation task (Section 4.1) and UEA classification task (Section 4.2).
- [2024/01/22] Support temporal point process task (Section 4.3).
 
## Environment Dependencies

`ContiFormer` is currently part of "[PhysioPro](https://github.com/microsoft/physiopro)" project, please first clone `PhysioPro` repo and set up the required environment.

## Reproduce the experimental result for interpolating continuous-time function

Run the following command, where `cc` controls the type of spiral, `0` for the first type, `1` for the second type, `2` for mixture (see Appendix C.1.1 for more information).

* For Neural ODE, please run

```
python spiral.py --adjoint=1 --visualize=1 --niters=10000 --model_name Neural_ODE --noise_a=0.02 --cc=2 --train_dir ./sprial_neuralode
```

* For ContiFormer, please run

```
python spiral.py --adjoint=1 --visualize=1 --niters=10000 --model_name Contiformer --noise_a=0.02 --cc=2 --train_dir ./spiral_contiformer
```

The results and visualization data will be saved to `./sprial_neuralode` and `./spiral_contiformer`. 

All the experimental results are averaged, i.e., `27, 42, 1024` (the same random seeds for other tasks), use `--seed` to set the seed.


## Reproduce the experimental result for irregular time series classification

1. Download the dataset

```bash
cd PhysioPro
mkdir data
wget http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip -P data

unzip data/Multivariate2018_ts.zip -d data/
rm data/Multivariate2018_ts.zip
```

2. Run irregular time series classification task with `ContiFormer` and `Heartbeat` dataset.

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

To obtain the result for `Heartbeat` dataset with 0.3 mask ratio under different random seeds, please run the following three commands.

```
python -m physiopro.entry.train docs/configs/contiformer_mask_classification.yml --data.mask_ratio 0.3 --data.name Heartbeat --runtime.seed 27
python -m physiopro.entry.train docs/configs/contiformer_mask_classification.yml --data.mask_ratio 0.3 --data.name Heartbeat --runtime.seed 42
python -m physiopro.entry.train docs/configs/contiformer_mask_classification.yml --data.mask_ratio 0.3 --data.name Heartbeat --runtime.seed 1024
```

#### Hyper-parameter search

We provided hyper-parameter searching for all the compared methods. Taking ContiFormer model as an example, we search activation function in `sigmoid` and `tanh` (please refer to Appendix D.4 for more information), and report the best performance for each dataset.

To perform hyper-parameter search for the activation function, please run the following commands:

```
python -m physiopro.entry.train docs/configs/contiformer_mask_classification.yml --data.mask_ratio 0.3 --data.name Heartbeat --network.actfn_ode sigmoid
python -m physiopro.entry.train docs/configs/contiformer_mask_classification.yml --data.mask_ratio 0.3 --data.name Heartbeat --network.actfn_ode tanh
```

## Reproduce the experimental result for temporal point process

1. Download the dataset

Please download the dataset from Google Drive [Link](https://drive.google.com/drive/folders/1SvHEiNuMH2lauQT5uYvNrdFoHi8ucSzx?usp=sharing), and put it under `data` fold.

2. Run temporal point process task with `ContiFormer` on Neonate dataset

```bash
# create the output directory
mkdir -p outputs/Temporal_Point_Process/neonate
# run the train task
python -m physiopro.entry.train docs/configs/contiformer_tpp.yml
# tensorboard
tensorboard --logdir outputs/
```

3. To change the fold, please add the following parameter

```bash
python -m physiopro.entry.train docs/configs/contiformer_tpp.yml --data.fold fold1
```

4. For other datasets, please run the following commands

```bash
python -m physiopro.entry.train docs/configs/contiformer_tpp.yml --data.name data_synthetic --model.lr 1e-2 --network.add_pe false --network.normalize_before false --network.actfn_ode sigmoid --network.layer_type_ode concatnorm --model.tmax 5 --model.step_size 100 --runtime.output_dir outputs/Temporal_Point_Process/synthetic
python -m physiopro.entry.train docs/configs/contiformer_tpp.yml --data.name data_mimic --model.lr 1e-3 --network.add_pe false --network.normalize_before true --network.actfn_ode sigmoid --network.layer_type_ode concatnorm --model.tmax 10 --model.step_size 20 --runtime.output_dir outputs/Temporal_Point_Process/mimic
python -m physiopro.entry.train docs/configs/contiformer_tpp.yml --data.name data_stackoverflow --model.lr 1e-3 --network.add_pe false --network.normalize_before false --network.actfn_ode sigmoid --network.layer_type_ode concat --model.tmax 10 --model.step_size 20  --runtime.output_dir outputs/Temporal_Point_Process/stackoverflow
python -m physiopro.entry.train docs/configs/contiformer_tpp.yml --data.name data_bookorder --model.lr 1e-3 --network.add_pe true --network.normalize_before false --network.actfn_ode sigmoid --network.layer_type_ode concatnorm --data.clip_max 70 --model.tmax 70 --model.step_size 20 --runtime.output_dir outputs/Temporal_Point_Process/bookorder
python -m physiopro.entry.train docs/configs/contiformer_tpp_neonate.yml --data.name data_neonate --model.lr 1e-2 --network.add_pe false --network.normalize_before false --network.actfn_ode tanh --network.layer_type_ode concat --model.tmax 20 --model.step_size 20  --runtime.output_dir outputs/Temporal_Point_Process/neonate
python -m physiopro.entry.train docs/configs/contiformer_tpp.yml --data.name data_traffic --model.lr 1e-3 --network.add_pe false --network.normalize_before false --network.actfn_ode sigmoid --network.layer_type_ode concat --model.tmax 5 --model.step_size 20  --runtime.output_dir outputs/Temporal_Point_Process/traffic
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
