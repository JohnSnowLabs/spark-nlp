---
layout: model
title: German roberta_base_wechsel_german_cimt_transport_pipeline pipeline RoBertaForSequenceClassification from juliaromberg
author: John Snow Labs
name: roberta_base_wechsel_german_cimt_transport_pipeline
date: 2025-02-07
tags: [de, open_source, pipeline, onnx]
task: Text Classification
language: de
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_base_wechsel_german_cimt_transport_pipeline` is a German model originally trained by juliaromberg.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_wechsel_german_cimt_transport_pipeline_de_5.5.1_3.0_1738898397624.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_wechsel_german_cimt_transport_pipeline_de_5.5.1_3.0_1738898397624.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_base_wechsel_german_cimt_transport_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_base_wechsel_german_cimt_transport_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_wechsel_german_cimt_transport_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|468.0 MB|

## References

https://huggingface.co/juliaromberg/roberta-base-wechsel-german_cimt-transport

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification