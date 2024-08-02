---
layout: model
title: English t5_tiny_random_length_128_learning_rate_2e_05_weight_decay_0_01_finetuned_english_tonga_tonga_islands_romanian_pipeline pipeline T5Transformer from ffsouza
author: John Snow Labs
name: t5_tiny_random_length_128_learning_rate_2e_05_weight_decay_0_01_finetuned_english_tonga_tonga_islands_romanian_pipeline
date: 2024-08-02
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_tiny_random_length_128_learning_rate_2e_05_weight_decay_0_01_finetuned_english_tonga_tonga_islands_romanian_pipeline` is a English model originally trained by ffsouza.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_tiny_random_length_128_learning_rate_2e_05_weight_decay_0_01_finetuned_english_tonga_tonga_islands_romanian_pipeline_en_5.4.2_3.0_1722563474304.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_tiny_random_length_128_learning_rate_2e_05_weight_decay_0_01_finetuned_english_tonga_tonga_islands_romanian_pipeline_en_5.4.2_3.0_1722563474304.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_tiny_random_length_128_learning_rate_2e_05_weight_decay_0_01_finetuned_english_tonga_tonga_islands_romanian_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_tiny_random_length_128_learning_rate_2e_05_weight_decay_0_01_finetuned_english_tonga_tonga_islands_romanian_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_tiny_random_length_128_learning_rate_2e_05_weight_decay_0_01_finetuned_english_tonga_tonga_islands_romanian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|24.1 MB|

## References

https://huggingface.co/ffsouza/t5-tiny-random-length-128-learning_rate-2e-05-weight_decay-0.01-finetuned-en-to-ro

## Included Models

- DocumentAssembler
- T5Transformer