---
layout: model
title: English bert_finetuned_gesture_prediction_9_classes_pipeline pipeline BertForTokenClassification from qfrodicio
author: John Snow Labs
name: bert_finetuned_gesture_prediction_9_classes_pipeline
date: 2025-02-06
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_finetuned_gesture_prediction_9_classes_pipeline` is a English model originally trained by qfrodicio.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_finetuned_gesture_prediction_9_classes_pipeline_en_5.5.1_3.0_1738828384884.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_finetuned_gesture_prediction_9_classes_pipeline_en_5.5.1_3.0_1738828384884.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_finetuned_gesture_prediction_9_classes_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_finetuned_gesture_prediction_9_classes_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_finetuned_gesture_prediction_9_classes_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|403.7 MB|

## References

https://huggingface.co/qfrodicio/bert-finetuned-gesture-prediction-9-classes

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification