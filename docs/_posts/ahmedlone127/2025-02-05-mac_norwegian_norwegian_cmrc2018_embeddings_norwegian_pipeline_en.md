---
layout: model
title: English mac_norwegian_norwegian_cmrc2018_embeddings_norwegian_pipeline pipeline BertForQuestionAnswering from wskhanh
author: John Snow Labs
name: mac_norwegian_norwegian_cmrc2018_embeddings_norwegian_pipeline
date: 2025-02-05
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mac_norwegian_norwegian_cmrc2018_embeddings_norwegian_pipeline` is a English model originally trained by wskhanh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mac_norwegian_norwegian_cmrc2018_embeddings_norwegian_pipeline_en_5.5.1_3.0_1738789667780.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mac_norwegian_norwegian_cmrc2018_embeddings_norwegian_pipeline_en_5.5.1_3.0_1738789667780.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mac_norwegian_norwegian_cmrc2018_embeddings_norwegian_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mac_norwegian_norwegian_cmrc2018_embeddings_norwegian_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mac_norwegian_norwegian_cmrc2018_embeddings_norwegian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|381.1 MB|

## References

https://huggingface.co/wskhanh/Mac-no-no-cmrc2018-embeddings-no

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering