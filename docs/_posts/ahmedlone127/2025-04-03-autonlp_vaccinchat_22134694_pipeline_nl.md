---
layout: model
title: Dutch, Flemish autonlp_vaccinchat_22134694_pipeline pipeline RoBertaForSequenceClassification from maximedb
author: John Snow Labs
name: autonlp_vaccinchat_22134694_pipeline
date: 2025-04-03
tags: [nl, open_source, pipeline, onnx]
task: Text Classification
language: nl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`autonlp_vaccinchat_22134694_pipeline` is a Dutch, Flemish model originally trained by maximedb.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/autonlp_vaccinchat_22134694_pipeline_nl_5.5.1_3.0_1743694663336.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/autonlp_vaccinchat_22134694_pipeline_nl_5.5.1_3.0_1743694663336.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("autonlp_vaccinchat_22134694_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("autonlp_vaccinchat_22134694_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|autonlp_vaccinchat_22134694_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|438.5 MB|

## References

https://huggingface.co/maximedb/autonlp-vaccinchat-22134694

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification