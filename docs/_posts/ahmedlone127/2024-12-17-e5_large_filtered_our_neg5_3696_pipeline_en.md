---
layout: model
title: English e5_large_filtered_our_neg5_3696_pipeline pipeline E5Embeddings from seongil-dn
author: John Snow Labs
name: e5_large_filtered_our_neg5_3696_pipeline
date: 2024-12-17
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

Pretrained E5Embeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`e5_large_filtered_our_neg5_3696_pipeline` is a English model originally trained by seongil-dn.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/e5_large_filtered_our_neg5_3696_pipeline_en_5.5.1_3.0_1734397715814.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/e5_large_filtered_our_neg5_3696_pipeline_en_5.5.1_3.0_1734397715814.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("e5_large_filtered_our_neg5_3696_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("e5_large_filtered_our_neg5_3696_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|e5_large_filtered_our_neg5_3696_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|636.9 MB|

## References

https://huggingface.co/seongil-dn/e5-large-filtered-our-neg5-3696

## Included Models

- DocumentAssembler
- E5Embeddings