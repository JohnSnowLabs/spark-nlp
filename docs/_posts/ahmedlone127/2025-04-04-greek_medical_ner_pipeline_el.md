---
layout: model
title: Modern Greek (1453-) greek_medical_ner_pipeline pipeline BertForTokenClassification from HUMADEX
author: John Snow Labs
name: greek_medical_ner_pipeline
date: 2025-04-04
tags: [el, open_source, pipeline, onnx]
task: Named Entity Recognition
language: el
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`greek_medical_ner_pipeline` is a Modern Greek (1453-) model originally trained by HUMADEX.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/greek_medical_ner_pipeline_el_5.5.1_3.0_1743742713731.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/greek_medical_ner_pipeline_el_5.5.1_3.0_1743742713731.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("greek_medical_ner_pipeline", lang = "el")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("greek_medical_ner_pipeline", lang = "el")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|greek_medical_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|el|
|Size:|403.7 MB|

## References

https://huggingface.co/HUMADEX/greek_medical_ner

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification