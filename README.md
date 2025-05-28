# The official code for the metric framework PROBE

PROBE directory consists of 7 models and an implementation file for PROBE

## 7 models
The original source code for each model is as follows

- [ComplEx and RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
- [TuckEr](https://github.com/ibalazevic/TuckER)
- [HousE](https://github.com/rui9812/HousE)
- [pLogicNet](https://github.com/DeepGraphLearning/pLogicNet)
- [RNNLogic](https://github.com/DeepGraphLearning/RNNLogic)
- CompGCN : The [original code](https://github.com/malllabiisc/CompGCN) wasn't runable in our enviornment. Thus we implement our own CompGCN by using [pykeen](https://github.com/pykeen/pykeen).

Since the original embedding dimensions were too large (up to 1000), we lacked the computational resources to run the models. Therefore, we standardized all embedding dimensions to 128, except for those that were already below this threshold, and re-tuned the corresponding hyperparameters.

All requirements and best configurations are located in each model directory. Researchers or practitioners facing similar limitations are welcome to use our optimized configurations freely.

## probe.py
The python file contains the implementation of PROBE. Each of the 7 models use probe.py for evaluation along with their original evaluations(MR, MRR, Hits@k).

Since probe.py is outside the model directories, it is crucial to run the below line first before any other part of the code that is related to evaluation is executed.
<pre><code>export PYTHONPATH=$(pwd)/..
</code></pre>
Although this line is already specified inside every run.sh file, make sure it is present.

## Acknowledgement
We thank the original authors for their great works.