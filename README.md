# Mixed-Precision Neural Network Quantization via Learned Layer-wise Importance and Pruning(LIMPQP)

## 1. Abstract

We implemented Learned Step Size Quantization and trained a importance indicator to determine the optimal bit-width for each layer alongside Iterative Magnitude Pruning with Rewinding to optimize our neural network model.
y

## 2. Implementation Differences From the Original Repo
### Pruning
You can use the following command to apply pruning method during the pre-training phase:
```
cd indicators_pretraining && python pruning.py {CONFIG.YAML} 
```

### Initial Values of the Quantization Step Size

In the original repo, the step sizes in weight quantization layers are initialized as`Tensor(max((mean-3*std).abs(), (mean+3*std).abs())/2**(bit_width-1))` where mean is `x.detach().mean()` and std is `x.detach().std()`, and in activation quantization layers, the step sizes are initialized as `Tensor(1.0)`.

In my implementation, the step sizes in activation quantization layers are initialized in the same way, but in weight quantization layers, the step sizes are initialized as `Tensor(v.abs().mean() * 2 / sqrt(Qp))`.


## 3. Importance Indicators Pre-training (One-time Training for Importance Derivation)
Firstly, you can pre-train the importance indicators for your models, or you can also use our previous indicators (in quantization_search/indicators/importance_indicators_resnet50.pkl). 

### Pre-train the indicators

```
cd indicators_pretraining && python -m torch.distributed.launch --nproc_per_node={NUM_GPUs} main.py {CONFIG.YAML} 
```

You can find the template YAML configuration file in "indicators_pretraining/config_resnet50.yaml". 

Meanwhile, if you want to use your own PyTorch model, you should add it to the *create_model* function (see indicators_pretraining/model/model.py) and designate it in the YAML configuration file. 

### Some Tips 

- For indicators pretraining, please set the first layer, its batchnorm layer, and the last layer for exceptions (do not be quantized), since these layers are not searchable and are quantized to fixed bits (8bits) during fine-tuning. 

- Pre-training does not require too many epochs, and even does not rely on the full training set, you can try 3~10 epochs and 50% data. 

### Extract the indicators

You should extract the indicators from the checkpoint after pre-training, since these indicators are ***quantization step-size scale-factors*** —— some learnable PyTorch parameters. This is quite easy, since we can traverse the checkpoint and record all step-size scale factors accordingly. 

In pre-training, the quantization step-size scale-factor for each layer has a specific variable name in the weight/activation quantizer (see indicators_pretraining/quan/quantizer/lsq.py, LINE66). 

For example, for layer "*module.layer2.0.conv2*", its activation and weight indicators are named "*module.layer2.0.conv2.quan_a_fn.s*" and "*module.layer2.0.conv2.quan_w_fn.s*", respectively. That means you can access all indicators with these orderly variable names.  

**The indicator extractor example code is in "indicators_pretraining/importance_extractor.py".** 

## 4. ILP-based MPQ Policy Search

### Search with provided constraints

Our code provides two constraints: BitOPs and model size (compression ratio), and at least one constraint should be enabled. 

Once obtaining the indicators, you can perform constraint search several times using the same indicators with below args: 

| Args   | Description                                                  | Example      |
| ------ | ------------------------------------------------------------ | ------------ |
| --i    | path of the importance indicators obtained by "*importance_extractor.py*" | data/r50.pkl |
| --b    | bit-width candidates                                         | 6 5 4 3 2    |
| --wb   | expected weight bit-width                                    | 3            |
| --ab   | expected activation bit-width                                | 3            |
| --cr   | model compression ratio (CR) constraint, cr=0 means disable this constraint | 12.2         |
| --bops | use BitOPs as a constraint                                   | True/False   |

 As an example, one can use the following command:

```
python search.py --model resnet50 --i indicators/importance_indicators_resnet50.pkl --b 6 5 4 3 2 --wb 3 --ab 4 --cr 12.2 --bops True 
```

And you will get a MPQ policy immediately: 

```
searched weight bit-widths [6, 6, 3, 6, 3, 4, 3, 4, 3, 2, 2, 5, 3, 2, 3, 2, 3, 3, 3, 2, 3, 4, 5, 3, 2, 2, 2, 2, 2, 3, 3, 2, 3, 2, 2, 3, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
searched act bit-widths [6, 4, 6, 6, 6, 4, 6, 6, 5, 6, 6, 2, 6, 6, 6, 6, 6, 6, 4, 6, 6, 2, 5, 5, 3, 6, 6, 6, 4, 4, 6, 3, 5, 6, 3, 5, 6, 3, 4, 6, 3, 6, 3, 2, 5, 4, 6, 3, 4, 6, 3, 5]
```

### Additional constraints

You can easily add other constraints (such as on-device latency), please refer the code.  



## 5. Fine-tuning & Evaluation

#### Fine-tuning

You can use any quantization algorithms to finetune your model. 

In our paper, the quantization algorithm is LSQ, see "*quantization_training*" folder and "*quantization_training/config_resnet50.yaml*" for details. 

Please paste your MPQ policy to the YAML file and use conventional training script to finetune the model. You can start from the above searched ResNet50's MPQ policy through an example YAML file: 

```
cd quantization_training && python -m torch.distributed.launch --nproc_per_node={NUM_GPUs} main.py finetune_resnet50_w3a4_12.2compression_ratio.yaml
```

#### Evaluation

```
cd quantization_training && python -m torch.distributed.launch --nproc_per_node=2 main.py {YAML_FILE.YAML}
```

## 6. Experiments
At the current stage, we tested fine tuning using a single GPU on Windows and uploaded the results to [Wandb](https://wandb.ai/stevenli/LIMPQP).
  

## 7. Acknowledgement

The authors would like to thank the following insightful open-source projects & papers, this work cannot be done without all of them:

- LSQ implementation: https://github.com/zhutmost/lsq-net
- LIMPQ: https://github.com/1hunters/LIMPQ

## 7. Future Work
At present, pruning only applies to the importance indicators pre-training, and in the future, we will add pruning to the fine-tuning. Meanwhile, we will also conduct experiments on the Transformer model.

