---
layout: model
title: Hungarian reading_comprehension_hurc_mt5_hungarian_pipeline pipeline T5Transformer from NYTK
author: John Snow Labs
name: reading_comprehension_hurc_mt5_hungarian_pipeline
date: 2024-08-26
tags: [hu, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: hu
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`reading_comprehension_hurc_mt5_hungarian_pipeline` is a Hungarian model originally trained by NYTK.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/reading_comprehension_hurc_mt5_hungarian_pipeline_hu_5.4.2_3.0_1724648340675.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/reading_comprehension_hurc_mt5_hungarian_pipeline_hu_5.4.2_3.0_1724648340675.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("reading_comprehension_hurc_mt5_hungarian_pipeline", lang = "hu")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("reading_comprehension_hurc_mt5_hungarian_pipeline", lang = "hu")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|reading_comprehension_hurc_mt5_hungarian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|hu|
|Size:|2.4 GB|

## References

https://huggingface.co/NYTK/reading-comprehension-hurc-mt5-hungarian

## Included Models

- DocumentAssembler
- T5Transformer