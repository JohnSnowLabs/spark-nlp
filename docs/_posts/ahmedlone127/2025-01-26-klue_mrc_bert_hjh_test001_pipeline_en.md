---
layout: model
title: English klue_mrc_bert_hjh_test001_pipeline pipeline BertForQuestionAnswering from cooler8
author: John Snow Labs
name: klue_mrc_bert_hjh_test001_pipeline
date: 2025-01-26
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`klue_mrc_bert_hjh_test001_pipeline` is a English model originally trained by cooler8.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/klue_mrc_bert_hjh_test001_pipeline_en_5.5.1_3.0_1737918803178.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/klue_mrc_bert_hjh_test001_pipeline_en_5.5.1_3.0_1737918803178.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("klue_mrc_bert_hjh_test001_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("klue_mrc_bert_hjh_test001_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|klue_mrc_bert_hjh_test001_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|412.4 MB|

## References

https://huggingface.co/cooler8/klue-mrc-bert-hjh-test001

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering