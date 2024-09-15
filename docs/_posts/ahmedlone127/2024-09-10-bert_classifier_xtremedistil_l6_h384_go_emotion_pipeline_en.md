---
layout: model
title: English bert_classifier_xtremedistil_l6_h384_go_emotion_pipeline pipeline BertForSequenceClassification from bergum
author: John Snow Labs
name: bert_classifier_xtremedistil_l6_h384_go_emotion_pipeline
date: 2024-09-10
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_classifier_xtremedistil_l6_h384_go_emotion_pipeline` is a English model originally trained by bergum.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_xtremedistil_l6_h384_go_emotion_pipeline_en_5.5.0_3.0_1725957223529.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_xtremedistil_l6_h384_go_emotion_pipeline_en_5.5.0_3.0_1725957223529.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_classifier_xtremedistil_l6_h384_go_emotion_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_classifier_xtremedistil_l6_h384_go_emotion_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_xtremedistil_l6_h384_go_emotion_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|84.3 MB|

## References

https://huggingface.co/bergum/xtremedistil-l6-h384-go-emotion

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification