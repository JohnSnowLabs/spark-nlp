---
layout: docs
header: true
title: GPU vs CPU benchmark on ClassifierDL
permalink: /docs/en/CPUvsGPUbenchmarkClassifierDL
key: docs-concepts
modify_date: "2021-08-31"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

## GPU vs CPU benchmark on ClassifierDL

This experiment consisted of training a document level Deep Learning Binary Classifier (Question vs Statement classes) using a fully connected CNN and Bert Sentence Embeddings. Only 1 Spark node was used for the training.

We used the Spark NLP class `ClassifierDL` and its method `Approach()` as described in the [documentation](https://nlp.johnsnowlabs.com/docs/en/annotators#classifierdl).

The pipeline looks as follows:

![](/assets/images/gpu_vs_cpu_pic3.png)

</div>
<div class="h3-box" markdown="1">

## Machine specs

### CPU
An AWS `m5.8xlarge` machine was used for the CPU benchmarking. This machine consists of `32 vCPUs` and `128 GB of RAM`, as you can check in the official specification webpage available [here](https://aws.amazon.com/ec2/instance-types/m5/)

### GPU
A `Tesla V100 SXM2` GPU with `32GB` of memory was used to calculate the GPU benchmarking.

</div>
<div class="h3-box" markdown="1">

## Versions
The benchmarking was carried out with the following Spark NLP versions:

Spark version: `3.0.2`

Hadoop version: `3.2.0`

SparkNLP version: `3.1.2`

Spark nodes: 1

</div>
<div class="h3-box" markdown="1">

## Dataset
The size of the dataset was relatively small (200K), consisting of:

Training (rows): `162250`

Test (rows): `40301`

</div>
<div class="h3-box" markdown="1">

## Training params
Different batch sizes were tested to demonstrate how GPU performance improves with bigger batches compared to CPU, for a constant number of epochs and learning rate.

Epochs: `10`

Learning rate:  `0.003`

Batch sizes: `32`, `64`, `256`, `1024`

</div>
<div class="h3-box" markdown="1">

## Results
Even for this average-sized dataset, we can observe that GPU is able to beat the CPU machine by an `19% in training time`, and a `15% in inference time`.

### Training times depending on batch (in minutes)

![](/assets/images/gpu_vs_cpu_pic1.png)

{:.table-model-big}
| Batch size | CPU | GPU |
| :---: | :---: | :--: |
|  32  |  66  |  60  |
|  64  |  65  |  56  |
|  256  |  64  |  53  |
|  1024  |  64  |  52  |

We can see that CPU didn't scale with bigger batches, although GPU got improvement specially up to a 256 batch size, not providing much more improvement afterwards.

</div>
<div class="h3-box" markdown="1">

### Inference times (in minutes)
The average inference time remained more or less constant regardless the batch size:
CPU: 8.7 min
GPU: 7.4 min

![](/assets/images/gpu_vs_cpu_pic5.png)

</div>
<div class="h3-box" markdown="1">

### Performance metrics
A weighted F1-score of 0.88 was achieved, with a 0.90 score for question detection and 0.83 for statements.

![](/assets/images/gpu_vs_cpu_pic2.png)

</div>
<div class="h3-box" markdown="1">

## Takeaways
The main takeaways of this benchmark were:
- GPU provides with an improvement in performance even for relatively small datasets
- GPU performance scales with bigger batch sizes compared to almost constant CPU times
- We achieve bigger improvement in training stage than in inference, although both times are better than in CPU

</div>
<div class="h3-box" markdown="1">

## How to get the best of the GPU
Not always GPU will beat CPU machines. Even in our use case, where we got almost a 20% of improvement, the GPU remains only at 40% of its utilization, what shows there is still room for improvement:

![](/assets/images/gpu_vs_cpu_pic4.png)

The level of improvement depends on the specs on both machines and, evidently, on the model to be trained and the size of the dataset.

Bigger improvements can be achived with bigger datasets and big batch sizes.

Also, pipelines with heavy computational cost, as `SentenceEntityResolver` (see Training documentation [here](https://nlp.johnsnowlabs.com/docs/en/licensed_training) are more suitable to achieve improvements using a GPU.

</div>
<div class="h3-box" markdown="1">

## Where to look for more information about Training
Please, take a look at the [Spark NLP](https://nlp.johnsnowlabs.com/docs/en/training) and [Spark NLP for Healthcare](https://nlp.johnsnowlabs.com/docs/en/licensed_training) Training sections, and feel free to reach us out in case you want to maximize the performance on your GPU.

</div>