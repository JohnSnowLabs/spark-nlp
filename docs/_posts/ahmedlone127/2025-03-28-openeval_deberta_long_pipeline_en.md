---
layout: model
title: English openeval_deberta_long_pipeline pipeline DeBertaForSequenceClassification from jgilhuly16
author: John Snow Labs
name: openeval_deberta_long_pipeline
date: 2025-03-28
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`openeval_deberta_long_pipeline` is a English model originally trained by jgilhuly16.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/openeval_deberta_long_pipeline_en_5.5.1_3.0_1743130595210.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/openeval_deberta_long_pipeline_en_5.5.1_3.0_1743130595210.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("openeval_deberta_long_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("openeval_deberta_long_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|openeval_deberta_long_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|597.1 MB|

## References

https://huggingface.co/jgilhuly16/openeval-deberta-long

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification