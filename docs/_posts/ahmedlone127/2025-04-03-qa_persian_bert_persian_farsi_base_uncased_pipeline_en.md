---
layout: model
title: English qa_persian_bert_persian_farsi_base_uncased_pipeline pipeline BertForQuestionAnswering from makhataei
author: John Snow Labs
name: qa_persian_bert_persian_farsi_base_uncased_pipeline
date: 2025-04-03
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`qa_persian_bert_persian_farsi_base_uncased_pipeline` is a English model originally trained by makhataei.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/qa_persian_bert_persian_farsi_base_uncased_pipeline_en_5.5.1_3.0_1743647279054.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/qa_persian_bert_persian_farsi_base_uncased_pipeline_en_5.5.1_3.0_1743647279054.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("qa_persian_bert_persian_farsi_base_uncased_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("qa_persian_bert_persian_farsi_base_uncased_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|qa_persian_bert_persian_farsi_base_uncased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|606.1 MB|

## References

https://huggingface.co/makhataei/qa-persian-bert-fa-base-uncased

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering