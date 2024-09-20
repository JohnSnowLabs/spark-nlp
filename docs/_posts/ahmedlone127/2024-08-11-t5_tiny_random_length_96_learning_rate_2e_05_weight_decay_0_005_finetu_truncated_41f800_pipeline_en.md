---
layout: model
title: English t5_tiny_random_length_96_learning_rate_2e_05_weight_decay_0_005_finetu_truncated_41f800_pipeline pipeline T5Transformer from diegor2
author: John Snow Labs
name: t5_tiny_random_length_96_learning_rate_2e_05_weight_decay_0_005_finetu_truncated_41f800_pipeline
date: 2024-08-11
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_tiny_random_length_96_learning_rate_2e_05_weight_decay_0_005_finetu_truncated_41f800_pipeline` is a English model originally trained by diegor2.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_tiny_random_length_96_learning_rate_2e_05_weight_decay_0_005_finetu_truncated_41f800_pipeline_en_5.4.2_3.0_1723338798435.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_tiny_random_length_96_learning_rate_2e_05_weight_decay_0_005_finetu_truncated_41f800_pipeline_en_5.4.2_3.0_1723338798435.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_tiny_random_length_96_learning_rate_2e_05_weight_decay_0_005_finetu_truncated_41f800_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_tiny_random_length_96_learning_rate_2e_05_weight_decay_0_005_finetu_truncated_41f800_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_tiny_random_length_96_learning_rate_2e_05_weight_decay_0_005_finetu_truncated_41f800_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|24.1 MB|

## References

https://huggingface.co/diegor2/t5-tiny-random-length-96-learning_rate-2e-05-weight_decay-0.005-finetu-truncated-41f800

## Included Models

- DocumentAssembler
- T5Transformer