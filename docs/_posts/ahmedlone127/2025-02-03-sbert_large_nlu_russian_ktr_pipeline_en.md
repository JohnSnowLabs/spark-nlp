---
layout: model
title: English sbert_large_nlu_russian_ktr_pipeline pipeline BertForQuestionAnswering from YuryCHep
author: John Snow Labs
name: sbert_large_nlu_russian_ktr_pipeline
date: 2025-02-03
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sbert_large_nlu_russian_ktr_pipeline` is a English model originally trained by YuryCHep.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sbert_large_nlu_russian_ktr_pipeline_en_5.5.1_3.0_1738593812683.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sbert_large_nlu_russian_ktr_pipeline_en_5.5.1_3.0_1738593812683.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sbert_large_nlu_russian_ktr_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sbert_large_nlu_russian_ktr_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbert_large_nlu_russian_ktr_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.6 GB|

## References

https://huggingface.co/YuryCHep/sbert_large_nlu_ru_ktr

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering