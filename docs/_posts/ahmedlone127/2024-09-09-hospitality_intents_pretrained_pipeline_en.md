---
layout: model
title: English hospitality_intents_pretrained_pipeline pipeline RoBertaForSequenceClassification from WellaBanda
author: John Snow Labs
name: hospitality_intents_pretrained_pipeline
date: 2024-09-09
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hospitality_intents_pretrained_pipeline` is a English model originally trained by WellaBanda.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hospitality_intents_pretrained_pipeline_en_5.5.0_3.0_1725912385631.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hospitality_intents_pretrained_pipeline_en_5.5.0_3.0_1725912385631.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hospitality_intents_pretrained_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hospitality_intents_pretrained_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hospitality_intents_pretrained_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|430.2 MB|

## References

https://huggingface.co/WellaBanda/hospitality_intents_pretrained

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification