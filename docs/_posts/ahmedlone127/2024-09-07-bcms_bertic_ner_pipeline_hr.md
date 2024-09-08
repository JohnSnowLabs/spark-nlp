---
layout: model
title: Croatian bcms_bertic_ner_pipeline pipeline BertForTokenClassification from classla
author: John Snow Labs
name: bcms_bertic_ner_pipeline
date: 2024-09-07
tags: [hr, open_source, pipeline, onnx]
task: Named Entity Recognition
language: hr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bcms_bertic_ner_pipeline` is a Croatian model originally trained by classla.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bcms_bertic_ner_pipeline_hr_5.5.0_3.0_1725726753667.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bcms_bertic_ner_pipeline_hr_5.5.0_3.0_1725726753667.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bcms_bertic_ner_pipeline", lang = "hr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bcms_bertic_ner_pipeline", lang = "hr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bcms_bertic_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|hr|
|Size:|412.8 MB|

## References

https://huggingface.co/classla/bcms-bertic-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification