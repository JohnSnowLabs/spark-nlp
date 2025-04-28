---
layout: model
title: English variome_0_0001_250_pipeline pipeline BertForTokenClassification from Brizape
author: John Snow Labs
name: variome_0_0001_250_pipeline
date: 2025-04-04
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`variome_0_0001_250_pipeline` is a English model originally trained by Brizape.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/variome_0_0001_250_pipeline_en_5.5.1_3.0_1743743674067.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/variome_0_0001_250_pipeline_en_5.5.1_3.0_1743743674067.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("variome_0_0001_250_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("variome_0_0001_250_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|variome_0_0001_250_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|408.3 MB|

## References

https://huggingface.co/Brizape/Variome_0.0001_250

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification