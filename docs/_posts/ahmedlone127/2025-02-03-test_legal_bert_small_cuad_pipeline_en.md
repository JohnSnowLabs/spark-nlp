---
layout: model
title: English test_legal_bert_small_cuad_pipeline pipeline BertForQuestionAnswering from kkkzzzkkk
author: John Snow Labs
name: test_legal_bert_small_cuad_pipeline
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`test_legal_bert_small_cuad_pipeline` is a English model originally trained by kkkzzzkkk.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/test_legal_bert_small_cuad_pipeline_en_5.5.1_3.0_1738594495181.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/test_legal_bert_small_cuad_pipeline_en_5.5.1_3.0_1738594495181.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("test_legal_bert_small_cuad_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("test_legal_bert_small_cuad_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|test_legal_bert_small_cuad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|130.6 MB|

## References

https://huggingface.co/kkkzzzkkk/test_legal-bert-small-cuad

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering