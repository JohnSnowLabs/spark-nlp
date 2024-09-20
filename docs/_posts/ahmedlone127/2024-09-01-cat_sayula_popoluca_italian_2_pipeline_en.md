---
layout: model
title: English cat_sayula_popoluca_italian_2_pipeline pipeline CamemBertForTokenClassification from homersimpson
author: John Snow Labs
name: cat_sayula_popoluca_italian_2_pipeline
date: 2024-09-01
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cat_sayula_popoluca_italian_2_pipeline` is a English model originally trained by homersimpson.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cat_sayula_popoluca_italian_2_pipeline_en_5.4.2_3.0_1725174928707.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cat_sayula_popoluca_italian_2_pipeline_en_5.4.2_3.0_1725174928707.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cat_sayula_popoluca_italian_2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cat_sayula_popoluca_italian_2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cat_sayula_popoluca_italian_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|391.7 MB|

## References

https://huggingface.co/homersimpson/cat-pos-it-2

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForTokenClassification