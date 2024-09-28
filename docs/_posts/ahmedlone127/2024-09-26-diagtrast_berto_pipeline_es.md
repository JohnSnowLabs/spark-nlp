---
layout: model
title: Castilian, Spanish diagtrast_berto_pipeline pipeline BertForSequenceClassification from somosnlp-hackathon-2023
author: John Snow Labs
name: diagtrast_berto_pipeline
date: 2024-09-26
tags: [es, open_source, pipeline, onnx]
task: Text Classification
language: es
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`diagtrast_berto_pipeline` is a Castilian, Spanish model originally trained by somosnlp-hackathon-2023.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/diagtrast_berto_pipeline_es_5.5.0_3.0_1727371233474.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/diagtrast_berto_pipeline_es_5.5.0_3.0_1727371233474.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("diagtrast_berto_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("diagtrast_berto_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|diagtrast_berto_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|411.7 MB|

## References

https://huggingface.co/somosnlp-hackathon-2023/DiagTrast-Berto

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification