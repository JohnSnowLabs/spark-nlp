---
layout: model
title: English legal_gqa_bert_augmented_1000_pipeline pipeline BertForQuestionAnswering from farid1088
author: John Snow Labs
name: legal_gqa_bert_augmented_1000_pipeline
date: 2024-09-23
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`legal_gqa_bert_augmented_1000_pipeline` is a English model originally trained by farid1088.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/legal_gqa_bert_augmented_1000_pipeline_en_5.5.0_3.0_1727050257283.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/legal_gqa_bert_augmented_1000_pipeline_en_5.5.0_3.0_1727050257283.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("legal_gqa_bert_augmented_1000_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("legal_gqa_bert_augmented_1000_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legal_gqa_bert_augmented_1000_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.9 MB|

## References

https://huggingface.co/farid1088/Legal_GQA_BERT_augmented_1000

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering