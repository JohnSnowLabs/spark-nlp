---
layout: model
title: Dutch, Flemish ul2_small_dutch_simplification_maithili_2023_pipeline pipeline T5Transformer from BramVanroy
author: John Snow Labs
name: ul2_small_dutch_simplification_maithili_2023_pipeline
date: 2024-08-06
tags: [nl, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: nl
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ul2_small_dutch_simplification_maithili_2023_pipeline` is a Dutch, Flemish model originally trained by BramVanroy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ul2_small_dutch_simplification_maithili_2023_pipeline_nl_5.4.2_3.0_1722918750937.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ul2_small_dutch_simplification_maithili_2023_pipeline_nl_5.4.2_3.0_1722918750937.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ul2_small_dutch_simplification_maithili_2023_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ul2_small_dutch_simplification_maithili_2023_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ul2_small_dutch_simplification_maithili_2023_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|349.6 MB|

## References

https://huggingface.co/BramVanroy/ul2-small-dutch-simplification-mai-2023

## Included Models

- DocumentAssembler
- T5Transformer