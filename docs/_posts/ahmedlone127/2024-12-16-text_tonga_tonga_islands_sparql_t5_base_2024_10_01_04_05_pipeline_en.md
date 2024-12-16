---
layout: model
title: English text_tonga_tonga_islands_sparql_t5_base_2024_10_01_04_05_pipeline pipeline T5Transformer from PopularPenguin
author: John Snow Labs
name: text_tonga_tonga_islands_sparql_t5_base_2024_10_01_04_05_pipeline
date: 2024-12-16
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`text_tonga_tonga_islands_sparql_t5_base_2024_10_01_04_05_pipeline` is a English model originally trained by PopularPenguin.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/text_tonga_tonga_islands_sparql_t5_base_2024_10_01_04_05_pipeline_en_5.5.1_3.0_1734327407524.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/text_tonga_tonga_islands_sparql_t5_base_2024_10_01_04_05_pipeline_en_5.5.1_3.0_1734327407524.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("text_tonga_tonga_islands_sparql_t5_base_2024_10_01_04_05_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("text_tonga_tonga_islands_sparql_t5_base_2024_10_01_04_05_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|text_tonga_tonga_islands_sparql_t5_base_2024_10_01_04_05_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.0 GB|

## References

https://huggingface.co/PopularPenguin/text-to-sparql-t5-base-2024-10-01_04-05

## Included Models

- DocumentAssembler
- T5Transformer