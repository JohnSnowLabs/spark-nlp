---
layout: model
title: English philai_embed_bge_test_pipeline pipeline BGEEmbeddings from dbourget
author: John Snow Labs
name: philai_embed_bge_test_pipeline
date: 2024-06-09
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BGEEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`philai_embed_bge_test_pipeline` is a English model originally trained by dbourget.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/philai_embed_bge_test_pipeline_en_5.4.0_3.0_1717959215055.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/philai_embed_bge_test_pipeline_en_5.4.0_3.0_1717959215055.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("philai_embed_bge_test_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("philai_embed_bge_test_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|philai_embed_bge_test_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.2 GB|

## References

https://huggingface.co/dbourget/philai-embed-bge-test

## Included Models

- DocumentAssembler
- BGEEmbeddings