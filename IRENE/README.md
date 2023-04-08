# Towards Inference Efficient Deep Ensemble Learning
This is the code for our AAAI 2023 paper "[Towards Inference Efficient Deep Ensemble Learning](https://seqml.github.io/irene/)".

## Abstract
> Ensemble methods can deliver surprising performance gains but also bring significantly higher computational costs, e.g., can be up to 2048X in large-scale ensemble tasks. However, we found that the majority of computations in ensemble methods are redundant. For instance, over 77% of samples in CIFAR-100 dataset can be correctly classified with only a single ResNet-18 model, which indicates that only around 23% of the samples need an ensemble of extra models. To this end, we propose an inference efficient ensemble learning method, to simultaneously optimize for effectiveness and efficiency in ensemble learning. More specifically, we regard ensemble of models as a sequential inference process and learn the optimal halting event for inference on a specific sample. At each timestep of the inference process, a common selector judges if the current ensemble has reached ensemble effectiveness and halt further inference, otherwise filters this challenging sample for the subsequent models to conduct more powerful ensemble. Both the base models and common selector are jointly optimized to dynamically adjust ensemble inference for different samples with various hardness, through the novel optimization goals including sequential ensemble boosting and computation saving. The experiments with different backbones on real-world datasets illustrate our method can bring up to 56% inference cost reduction while maintaining comparable performance to full ensemble, achieving significantly better ensemble utility than other baselines.

## Environment Dependencies
### Dependencies
```
pip install -r requirements.txt
```

### Running
As an example of an ensemble of 3 available ResNet-18 models on CIFAR10, you can run the following command to build both effective and efficient ensemble:
```
python run.py -dataset 'cifar10' -net resnet18 -n_estimators 3
```

## Special credits to Ensemble PyTorch

We give special credits to [Ensemble PyTorch](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch), which provides very good training and evaluation pipeline codes for ensemble algorithms to develop our code. And we do not declare this twice in each file again.

## Reference
You are more than welcome to cite our paper:
```
@misc{li2023inference,
      title={Towards Inference Efficient Deep Ensemble Learning}, 
      author={Ziyue Li and Kan Ren and Yifan Yang and Xinyang Jiang and Yuqing Yang and Dongsheng Li},
      year={2023},
      eprint={2301.12378},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
