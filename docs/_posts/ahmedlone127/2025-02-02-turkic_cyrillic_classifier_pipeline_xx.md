---
layout: model
title: Multilingual turkic_cyrillic_classifier_pipeline pipeline BertForSequenceClassification from tatiana-merz
author: John Snow Labs
name: turkic_cyrillic_classifier_pipeline
date: 2025-02-02
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`turkic_cyrillic_classifier_pipeline` is a Multilingual model originally trained by tatiana-merz.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/turkic_cyrillic_classifier_pipeline_xx_5.5.1_3.0_1738485300845.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/turkic_cyrillic_classifier_pipeline_xx_5.5.1_3.0_1738485300845.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("turkic_cyrillic_classifier_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("turkic_cyrillic_classifier_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|turkic_cyrillic_classifier_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|667.3 MB|

## References

https://huggingface.co/tatiana-merz/turkic-cyrillic-classifier

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification