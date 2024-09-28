---
layout: model
title: English aes_enem_models_sourcea_regression_from_bertimbau_large_c5_pipeline pipeline BertForSequenceClassification from kamel-usp
author: John Snow Labs
name: aes_enem_models_sourcea_regression_from_bertimbau_large_c5_pipeline
date: 2024-09-25
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`aes_enem_models_sourcea_regression_from_bertimbau_large_c5_pipeline` is a English model originally trained by kamel-usp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/aes_enem_models_sourcea_regression_from_bertimbau_large_c5_pipeline_en_5.5.0_3.0_1727261720430.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/aes_enem_models_sourcea_regression_from_bertimbau_large_c5_pipeline_en_5.5.0_3.0_1727261720430.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("aes_enem_models_sourcea_regression_from_bertimbau_large_c5_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("aes_enem_models_sourcea_regression_from_bertimbau_large_c5_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|aes_enem_models_sourcea_regression_from_bertimbau_large_c5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/kamel-usp/aes_enem_models-sourceA-regression-from-bertimbau-large-C5

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification