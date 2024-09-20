---
layout: model
title: English task_implicit_task__model_deberta__aug_method_gm_pipeline pipeline DeBertaForSequenceClassification from BenjaminOcampo
author: John Snow Labs
name: task_implicit_task__model_deberta__aug_method_gm_pipeline
date: 2024-08-31
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`task_implicit_task__model_deberta__aug_method_gm_pipeline` is a English model originally trained by BenjaminOcampo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/task_implicit_task__model_deberta__aug_method_gm_pipeline_en_5.4.2_3.0_1725117385169.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/task_implicit_task__model_deberta__aug_method_gm_pipeline_en_5.4.2_3.0_1725117385169.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("task_implicit_task__model_deberta__aug_method_gm_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("task_implicit_task__model_deberta__aug_method_gm_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|task_implicit_task__model_deberta__aug_method_gm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|607.5 MB|

## References

https://huggingface.co/BenjaminOcampo/task-implicit_task__model-deberta__aug_method-gm

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification