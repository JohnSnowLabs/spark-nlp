---
layout: model
title: English moderationbert_english_02_pipeline pipeline BertForSequenceClassification from ifmain
author: John Snow Labs
name: moderationbert_english_02_pipeline
date: 2025-03-27
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`moderationbert_english_02_pipeline` is a English model originally trained by ifmain.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/moderationbert_english_02_pipeline_en_5.5.1_3.0_1743090224697.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/moderationbert_english_02_pipeline_en_5.5.1_3.0_1743090224697.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("moderationbert_english_02_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("moderationbert_english_02_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|moderationbert_english_02_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|667.3 MB|

## References

https://huggingface.co/ifmain/ModerationBERT-En-02

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification