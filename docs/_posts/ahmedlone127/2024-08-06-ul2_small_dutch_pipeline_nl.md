---
layout: model
title: Dutch, Flemish ul2_small_dutch_pipeline pipeline T5Transformer from yhavinga
author: John Snow Labs
name: ul2_small_dutch_pipeline
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ul2_small_dutch_pipeline` is a Dutch, Flemish model originally trained by yhavinga.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ul2_small_dutch_pipeline_nl_5.4.2_3.0_1722910439434.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ul2_small_dutch_pipeline_nl_5.4.2_3.0_1722910439434.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ul2_small_dutch_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ul2_small_dutch_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ul2_small_dutch_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|349.7 MB|

## References

https://huggingface.co/yhavinga/ul2-small-dutch

## Included Models

- DocumentAssembler
- T5Transformer