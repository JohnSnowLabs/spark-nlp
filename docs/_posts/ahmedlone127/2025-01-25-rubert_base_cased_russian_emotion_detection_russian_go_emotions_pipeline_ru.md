---
layout: model
title: Russian rubert_base_cased_russian_emotion_detection_russian_go_emotions_pipeline pipeline BertForSequenceClassification from seara
author: John Snow Labs
name: rubert_base_cased_russian_emotion_detection_russian_go_emotions_pipeline
date: 2025-01-25
tags: [ru, open_source, pipeline, onnx]
task: Text Classification
language: ru
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`rubert_base_cased_russian_emotion_detection_russian_go_emotions_pipeline` is a Russian model originally trained by seara.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/rubert_base_cased_russian_emotion_detection_russian_go_emotions_pipeline_ru_5.5.1_3.0_1737840600545.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/rubert_base_cased_russian_emotion_detection_russian_go_emotions_pipeline_ru_5.5.1_3.0_1737840600545.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("rubert_base_cased_russian_emotion_detection_russian_go_emotions_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("rubert_base_cased_russian_emotion_detection_russian_go_emotions_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rubert_base_cased_russian_emotion_detection_russian_go_emotions_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|666.6 MB|

## References

https://huggingface.co/seara/rubert-base-cased-russian-emotion-detection-ru-go-emotions

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification