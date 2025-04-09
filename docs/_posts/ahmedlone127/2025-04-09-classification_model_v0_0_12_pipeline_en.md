---
layout: model
title: English classification_model_v0_0_12_pipeline pipeline BertForSequenceClassification from procit007
author: John Snow Labs
name: classification_model_v0_0_12_pipeline
date: 2025-04-09
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`classification_model_v0_0_12_pipeline` is a English model originally trained by procit007.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classification_model_v0_0_12_pipeline_en_5.5.1_3.0_1744165536488.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classification_model_v0_0_12_pipeline_en_5.5.1_3.0_1744165536488.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("classification_model_v0_0_12_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("classification_model_v0_0_12_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classification_model_v0_0_12_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/procit007/Classification_Model_v0.0.12

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification