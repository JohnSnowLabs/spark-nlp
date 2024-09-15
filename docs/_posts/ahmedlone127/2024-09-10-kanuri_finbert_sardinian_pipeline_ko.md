---
layout: model
title: Korean kanuri_finbert_sardinian_pipeline pipeline BertForSequenceClassification from snunlp
author: John Snow Labs
name: kanuri_finbert_sardinian_pipeline
date: 2024-09-10
tags: [ko, open_source, pipeline, onnx]
task: Text Classification
language: ko
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kanuri_finbert_sardinian_pipeline` is a Korean model originally trained by snunlp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kanuri_finbert_sardinian_pipeline_ko_5.5.0_3.0_1725977604210.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kanuri_finbert_sardinian_pipeline_ko_5.5.0_3.0_1725977604210.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kanuri_finbert_sardinian_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kanuri_finbert_sardinian_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kanuri_finbert_sardinian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|380.1 MB|

## References

https://huggingface.co/snunlp/KR-FinBert-SC

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification