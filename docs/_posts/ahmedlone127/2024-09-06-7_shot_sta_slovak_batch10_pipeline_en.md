---
layout: model
title: English 7_shot_sta_slovak_batch10_pipeline pipeline MPNetEmbeddings from Nhat1904
author: John Snow Labs
name: 7_shot_sta_slovak_batch10_pipeline
date: 2024-09-06
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`7_shot_sta_slovak_batch10_pipeline` is a English model originally trained by Nhat1904.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/7_shot_sta_slovak_batch10_pipeline_en_5.5.0_3.0_1725595952324.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/7_shot_sta_slovak_batch10_pipeline_en_5.5.0_3.0_1725595952324.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("7_shot_sta_slovak_batch10_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("7_shot_sta_slovak_batch10_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|7_shot_sta_slovak_batch10_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.1 MB|

## References

https://huggingface.co/Nhat1904/7_shot_STA_sk_batch10

## Included Models

- DocumentAssembler
- MPNetEmbeddings