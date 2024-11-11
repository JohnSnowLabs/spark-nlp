---
layout: model
title: German popbert_pipeline pipeline BertForSequenceClassification from luerhard
author: John Snow Labs
name: popbert_pipeline
date: 2024-11-11
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`popbert_pipeline` is a German model originally trained by luerhard.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/popbert_pipeline_de_5.5.1_3.0_1731310008414.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/popbert_pipeline_de_5.5.1_3.0_1731310008414.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("popbert_pipeline", lang = "de")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("popbert_pipeline", lang = "de")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|popbert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|1.3 GB|

## References

References

https://huggingface.co/luerhard/PopBERT

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification