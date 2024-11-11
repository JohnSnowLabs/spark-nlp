---
layout: model
title: Arabic propaganda_ner_arabic_pipeline pipeline BertForTokenClassification from ashrafulparan
author: John Snow Labs
name: propaganda_ner_arabic_pipeline
date: 2024-11-11
tags: [ar, open_source, pipeline, onnx]
task: Named Entity Recognition
language: ar
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`propaganda_ner_arabic_pipeline` is a Arabic model originally trained by ashrafulparan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/propaganda_ner_arabic_pipeline_ar_5.5.1_3.0_1731290544801.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/propaganda_ner_arabic_pipeline_ar_5.5.1_3.0_1731290544801.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("propaganda_ner_arabic_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("propaganda_ner_arabic_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|propaganda_ner_arabic_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|406.8 MB|

## References

https://huggingface.co/ashrafulparan/Propaganda-NER-Arabic

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification