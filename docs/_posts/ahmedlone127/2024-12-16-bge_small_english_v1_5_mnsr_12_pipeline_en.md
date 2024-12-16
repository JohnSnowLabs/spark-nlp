---
layout: model
title: English bge_small_english_v1_5_mnsr_12_pipeline pipeline BGEEmbeddings from jebish7
author: John Snow Labs
name: bge_small_english_v1_5_mnsr_12_pipeline
date: 2024-12-16
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

Pretrained BGEEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bge_small_english_v1_5_mnsr_12_pipeline` is a English model originally trained by jebish7.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bge_small_english_v1_5_mnsr_12_pipeline_en_5.5.1_3.0_1734358687732.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bge_small_english_v1_5_mnsr_12_pipeline_en_5.5.1_3.0_1734358687732.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bge_small_english_v1_5_mnsr_12_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bge_small_english_v1_5_mnsr_12_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bge_small_english_v1_5_mnsr_12_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|114.0 MB|

## References

https://huggingface.co/jebish7/bge-small-en-v1.5_MNSR_12

## Included Models

- DocumentAssembler
- BGEEmbeddings