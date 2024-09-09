---
layout: model
title: English turkish_bert_6493_5411_5464_v3_pipeline pipeline BertForSequenceClassification from muhammtcelik
author: John Snow Labs
name: turkish_bert_6493_5411_5464_v3_pipeline
date: 2024-09-08
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`turkish_bert_6493_5411_5464_v3_pipeline` is a English model originally trained by muhammtcelik.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/turkish_bert_6493_5411_5464_v3_pipeline_en_5.5.0_3.0_1725819613832.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/turkish_bert_6493_5411_5464_v3_pipeline_en_5.5.0_3.0_1725819613832.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("turkish_bert_6493_5411_5464_v3_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("turkish_bert_6493_5411_5464_v3_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|turkish_bert_6493_5411_5464_v3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|415.5 MB|

## References

https://huggingface.co/muhammtcelik/turkish-bert-6493-5411-5464_v3

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification