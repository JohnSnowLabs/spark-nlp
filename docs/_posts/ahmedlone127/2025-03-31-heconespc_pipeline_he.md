---
layout: model
title: Hebrew heconespc_pipeline pipeline RoBertaForSequenceClassification from HeTree
author: John Snow Labs
name: heconespc_pipeline
date: 2025-03-31
tags: [he, open_source, pipeline, onnx]
task: Text Classification
language: he
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`heconespc_pipeline` is a Hebrew model originally trained by HeTree.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/heconespc_pipeline_he_5.5.1_3.0_1743439656766.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/heconespc_pipeline_he_5.5.1_3.0_1743439656766.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("heconespc_pipeline", lang = "he")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("heconespc_pipeline", lang = "he")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|heconespc_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|he|
|Size:|468.1 MB|

## References

https://huggingface.co/HeTree/HeConEspc

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification