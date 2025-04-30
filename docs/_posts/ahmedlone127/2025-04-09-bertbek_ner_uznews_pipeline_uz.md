---
layout: model
title: Uzbek bertbek_ner_uznews_pipeline pipeline BertForTokenClassification from elmurod1202
author: John Snow Labs
name: bertbek_ner_uznews_pipeline
date: 2025-04-09
tags: [uz, open_source, pipeline, onnx]
task: Named Entity Recognition
language: uz
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bertbek_ner_uznews_pipeline` is a Uzbek model originally trained by elmurod1202.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bertbek_ner_uznews_pipeline_uz_5.5.1_3.0_1744199882630.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bertbek_ner_uznews_pipeline_uz_5.5.1_3.0_1744199882630.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bertbek_ner_uznews_pipeline", lang = "uz")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bertbek_ner_uznews_pipeline", lang = "uz")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bertbek_ner_uznews_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|uz|
|Size:|405.6 MB|

## References

https://huggingface.co/elmurod1202/bertbek-ner-uznews

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification