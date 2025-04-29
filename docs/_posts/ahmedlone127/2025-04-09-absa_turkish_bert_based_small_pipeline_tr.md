---
layout: model
title: Turkish absa_turkish_bert_based_small_pipeline pipeline BertForSequenceClassification from Sengil
author: John Snow Labs
name: absa_turkish_bert_based_small_pipeline
date: 2025-04-09
tags: [tr, open_source, pipeline, onnx]
task: Text Classification
language: tr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`absa_turkish_bert_based_small_pipeline` is a Turkish model originally trained by Sengil.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/absa_turkish_bert_based_small_pipeline_tr_5.5.1_3.0_1744224986892.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/absa_turkish_bert_based_small_pipeline_tr_5.5.1_3.0_1744224986892.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("absa_turkish_bert_based_small_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("absa_turkish_bert_based_small_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|absa_turkish_bert_based_small_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|414.5 MB|

## References

https://huggingface.co/Sengil/ABSA-Turkish-bert-based-small

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification