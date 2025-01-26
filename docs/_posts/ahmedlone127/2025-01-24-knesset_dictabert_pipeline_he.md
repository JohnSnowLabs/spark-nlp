---
layout: model
title: Hebrew knesset_dictabert_pipeline pipeline BertEmbeddings from GiliGold
author: John Snow Labs
name: knesset_dictabert_pipeline
date: 2025-01-24
tags: [he, open_source, pipeline, onnx]
task: Embeddings
language: he
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`knesset_dictabert_pipeline` is a Hebrew model originally trained by GiliGold.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/knesset_dictabert_pipeline_he_5.5.1_3.0_1737707939916.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/knesset_dictabert_pipeline_he_5.5.1_3.0_1737707939916.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("knesset_dictabert_pipeline", lang = "he")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("knesset_dictabert_pipeline", lang = "he")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|knesset_dictabert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|he|
|Size:|689.1 MB|

## References

https://huggingface.co/GiliGold/Knesset-DictaBERT

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings