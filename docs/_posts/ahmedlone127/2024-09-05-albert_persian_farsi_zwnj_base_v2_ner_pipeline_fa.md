---
layout: model
title: Persian albert_persian_farsi_zwnj_base_v2_ner_pipeline pipeline BertForTokenClassification from HooshvareLab
author: John Snow Labs
name: albert_persian_farsi_zwnj_base_v2_ner_pipeline
date: 2024-09-05
tags: [fa, open_source, pipeline, onnx]
task: Named Entity Recognition
language: fa
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albert_persian_farsi_zwnj_base_v2_ner_pipeline` is a Persian model originally trained by HooshvareLab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_persian_farsi_zwnj_base_v2_ner_pipeline_fa_5.5.0_3.0_1725511386044.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_persian_farsi_zwnj_base_v2_ner_pipeline_fa_5.5.0_3.0_1725511386044.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("albert_persian_farsi_zwnj_base_v2_ner_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("albert_persian_farsi_zwnj_base_v2_ner_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_persian_farsi_zwnj_base_v2_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|41.9 MB|

## References

https://huggingface.co/HooshvareLab/albert-fa-zwnj-base-v2-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification