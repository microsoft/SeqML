# SIMPLE: Specialized Model-Sample Matching for Domain Generalization
This repository contains the code for the paper titled "[SIMPLE: Specialized Model-Sample Matching for Domain Generalization](https://seqml.github.io/simple/)" that will be presented at the International Conference on Learning Representations (ICLR) 2023.

## Abstract
> In domain generalization (DG), most existing methods aspire to fine-tune a specific pretrained model through novel DG algorithms. In this paper, we propose an alternative direction, i.e., to efficiently leverage a pool of pretrained models without fine-tuning. Through extensive empirical and theoretical evidence, we demonstrate that (1) pretrained models have possessed generalization to some extent while there is no single best pretrained model across all distribution shifts, and (2) out-of-distribution (OOD) generalization error depends on the fitness between the pretrained model and unseen test distributions. This analysis motivates us to incorporate diverse pretrained models and to dispatch the best matched models for each OOD sample by means of recommendation techniques. To this end, we propose SIMPLE, a specialized model-sample matching method for domain generalization. First, the predictions of pretrained models are adapted to the target domain by a linear label space transformation. A matching network aware of model specialty is then proposed to dynamically recommend proper pretrained models to predict each test sample. The experiments on DomainBed show that our method achieves significant performance improvements (up to 12.2% for individual dataset and 3.9% on average) compared to state-of-the-art (SOTA) methods and further achieves 6.1% gain via enlarging the pretrained model pool. Moreover, our method is highly efficient and achieves more than 1000 times training speedup compared to the conventional DG methods with fine-tuning a pretrained model.

## Environment Dependencies
### Dependencies
The following command installs all the necessary dependencies for this repository:
```
pip install -r requirements.txt
```


### Download datasets
Run the following command to download the datasets used in the experiments:
```
python -m miscellaneous.domainbed.scripts.download --data_dir=./miscellaneous/domainbed/data
```

### Saving model predictions

Note that all pretrained models are not fine-tuned, so saving their predictions rather than take inference when needed can save a lot of time. Follow the steps given below to save the predictions of the pretrained models:

**Step 1:** Define the model pool by creating a .txt file that lists the names of the pre-trained models. This repository provides two examples: *'sample_list.txt'* and *'full_list.txt'*. 


**Step 2:** Set the path where the pretrained model cache will be saved to by running the following command:
```
export TORCH_HOME="./pytorch_pretrained_models/"
```


**Step 3:** Run the spec.py file with the following parameters to generate the predictions of all of the pretrained models in the model pool defined by the pretrain_model_list:
```
python -u spec.py --save_inference_only --dataset domainbed --domainbed_dataset PACS --pretrain_model_list modelpool_list/sample_list.txt --batch_size 256
```


### Running
To learn the specialized model-sample matching, use the following command as an example for the PACS dataset in DomainBed:
```
CUDA_VISIBLE_DEVICES=0 python spec.py --pretrain_model_list modelpool_list/sample_list.txt --dataset domainbed --domainbed_dataset PACS --domainbed_test_env 0
```

## Special credits to thirdparty repositories, such as DomainBed

Special credits are given to third-party repositories like [DomainBed](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch) for providing the necessary code to load datasets or neural networks. And we do not declare this in each file within the *'miscellaneous'* repository.

## Reference
You are more than welcome to cite our paper:
```
@inproceedings{lisimple,
  title={SIMPLE: Specialized Model-Sample Matching for Domain Generalization},
  author={Li, Ziyue and Ren, Kan and Jiang, Xinyang and Shen, Yifei and Zhang, Haipeng and Li, Dongsheng},
  booktitle={The Eleventh International Conference on Learning Representations}
}