---
layout: model
title: English perspective_conditional_utilitarian_deberta_01_pipeline pipeline DeBertaForSequenceClassification from edmundmills
author: John Snow Labs
name: perspective_conditional_utilitarian_deberta_01_pipeline
date: 2024-09-06
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`perspective_conditional_utilitarian_deberta_01_pipeline` is a English model originally trained by edmundmills.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/perspective_conditional_utilitarian_deberta_01_pipeline_en_5.5.0_3.0_1725611703221.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/perspective_conditional_utilitarian_deberta_01_pipeline_en_5.5.0_3.0_1725611703221.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("perspective_conditional_utilitarian_deberta_01_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("perspective_conditional_utilitarian_deberta_01_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|perspective_conditional_utilitarian_deberta_01_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.5 GB|

## References

https://huggingface.co/edmundmills/perspective-conditional-utilitarian-deberta-01

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification