---
layout: model
title: English t5_small_awesome_text_tonga_tonga_islands_sql_2024_11_10_13_40_pipeline pipeline T5Transformer from PopularPenguin
author: John Snow Labs
name: t5_small_awesome_text_tonga_tonga_islands_sql_2024_11_10_13_40_pipeline
date: 2025-01-29
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_small_awesome_text_tonga_tonga_islands_sql_2024_11_10_13_40_pipeline` is a English model originally trained by PopularPenguin.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_small_awesome_text_tonga_tonga_islands_sql_2024_11_10_13_40_pipeline_en_5.5.1_3.0_1738148652144.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_small_awesome_text_tonga_tonga_islands_sql_2024_11_10_13_40_pipeline_en_5.5.1_3.0_1738148652144.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_small_awesome_text_tonga_tonga_islands_sql_2024_11_10_13_40_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_small_awesome_text_tonga_tonga_islands_sql_2024_11_10_13_40_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_small_awesome_text_tonga_tonga_islands_sql_2024_11_10_13_40_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|288.9 MB|

## References

https://huggingface.co/PopularPenguin/t5-small-awesome-text-to-sql-2024-11-10_13-40

## Included Models

- DocumentAssembler
- T5Transformer