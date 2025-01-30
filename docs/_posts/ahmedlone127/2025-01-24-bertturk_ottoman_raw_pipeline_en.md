---
layout: model
title: English bertturk_ottoman_raw_pipeline pipeline BertEmbeddings from onurkeles
author: John Snow Labs
name: bertturk_ottoman_raw_pipeline
date: 2025-01-24
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bertturk_ottoman_raw_pipeline` is a English model originally trained by onurkeles.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bertturk_ottoman_raw_pipeline_en_5.5.1_3.0_1737742134102.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bertturk_ottoman_raw_pipeline_en_5.5.1_3.0_1737742134102.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bertturk_ottoman_raw_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bertturk_ottoman_raw_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bertturk_ottoman_raw_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|412.0 MB|

## References

https://huggingface.co/onurkeles/bertturk-ottoman-raw

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings