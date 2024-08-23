---
layout: model
title: English mt5_base_germeval21_toxic_with_task_specific_pretraining_and_data_augmentation_pipeline pipeline T5Transformer from airKlizz
author: John Snow Labs
name: mt5_base_germeval21_toxic_with_task_specific_pretraining_and_data_augmentation_pipeline
date: 2024-08-23
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_base_germeval21_toxic_with_task_specific_pretraining_and_data_augmentation_pipeline` is a English model originally trained by airKlizz.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_base_germeval21_toxic_with_task_specific_pretraining_and_data_augmentation_pipeline_en_5.4.2_3.0_1724387842510.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_base_germeval21_toxic_with_task_specific_pretraining_and_data_augmentation_pipeline_en_5.4.2_3.0_1724387842510.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_base_germeval21_toxic_with_task_specific_pretraining_and_data_augmentation_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_base_germeval21_toxic_with_task_specific_pretraining_and_data_augmentation_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_base_germeval21_toxic_with_task_specific_pretraining_and_data_augmentation_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|2.4 GB|

## References

https://huggingface.co/airKlizz/mt5-base-germeval21-toxic-with-task-specific-pretraining-and-data-augmentation

## Included Models

- DocumentAssembler
- T5Transformer