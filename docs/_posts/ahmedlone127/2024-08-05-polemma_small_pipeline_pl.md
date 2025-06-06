---
layout: model
title: Polish polemma_small_pipeline pipeline T5Transformer from amu-cai
author: John Snow Labs
name: polemma_small_pipeline
date: 2024-08-05
tags: [pl, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: pl
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`polemma_small_pipeline` is a Polish model originally trained by amu-cai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/polemma_small_pipeline_pl_5.4.2_3.0_1722901416120.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/polemma_small_pipeline_pl_5.4.2_3.0_1722901416120.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("polemma_small_pipeline", lang = "pl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("polemma_small_pipeline", lang = "pl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|polemma_small_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|pl|
|Size:|409.6 MB|

## References

https://huggingface.co/amu-cai/polemma-small

## Included Models

- DocumentAssembler
- T5Transformer