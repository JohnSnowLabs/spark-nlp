---
layout: model
title: Bulgarian punctual_bert_bulgarian_pipeline pipeline BertForTokenClassification from auhide
author: John Snow Labs
name: punctual_bert_bulgarian_pipeline
date: 2025-04-03
tags: [bg, open_source, pipeline, onnx]
task: Named Entity Recognition
language: bg
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`punctual_bert_bulgarian_pipeline` is a Bulgarian model originally trained by auhide.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/punctual_bert_bulgarian_pipeline_bg_5.5.1_3.0_1743715273856.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/punctual_bert_bulgarian_pipeline_bg_5.5.1_3.0_1743715273856.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("punctual_bert_bulgarian_pipeline", lang = "bg")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("punctual_bert_bulgarian_pipeline", lang = "bg")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|punctual_bert_bulgarian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|bg|
|Size:|665.1 MB|

## References

https://huggingface.co/auhide/punctual-bert-bg

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification