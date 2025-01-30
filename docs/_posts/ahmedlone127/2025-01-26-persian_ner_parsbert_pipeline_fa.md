---
layout: model
title: Persian persian_ner_parsbert_pipeline pipeline BertForTokenClassification from PardisSzah
author: John Snow Labs
name: persian_ner_parsbert_pipeline
date: 2025-01-26
tags: [fa, open_source, pipeline, onnx]
task: Named Entity Recognition
language: fa
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`persian_ner_parsbert_pipeline` is a Persian model originally trained by PardisSzah.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/persian_ner_parsbert_pipeline_fa_5.5.1_3.0_1737933806768.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/persian_ner_parsbert_pipeline_fa_5.5.1_3.0_1737933806768.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("persian_ner_parsbert_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("persian_ner_parsbert_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|persian_ner_parsbert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|606.5 MB|

## References

https://huggingface.co/PardisSzah/Persian_NER_parsbert

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification