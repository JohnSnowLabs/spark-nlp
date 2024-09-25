---
layout: model
title: English 2020_q2_50p_filtered_2_pipeline pipeline RoBertaEmbeddings from DouglasPontes
author: John Snow Labs
name: 2020_q2_50p_filtered_2_pipeline
date: 2024-09-23
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`2020_q2_50p_filtered_2_pipeline` is a English model originally trained by DouglasPontes.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/2020_q2_50p_filtered_2_pipeline_en_5.5.0_3.0_1727091929671.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/2020_q2_50p_filtered_2_pipeline_en_5.5.0_3.0_1727091929671.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("2020_q2_50p_filtered_2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("2020_q2_50p_filtered_2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|2020_q2_50p_filtered_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|463.9 MB|

## References

https://huggingface.co/DouglasPontes/2020-Q2-50p-filtered_2

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings