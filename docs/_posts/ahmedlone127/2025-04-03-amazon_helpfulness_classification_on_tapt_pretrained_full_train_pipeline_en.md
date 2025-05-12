---
layout: model
title: English amazon_helpfulness_classification_on_tapt_pretrained_full_train_pipeline pipeline RoBertaForSequenceClassification from ltuzova
author: John Snow Labs
name: amazon_helpfulness_classification_on_tapt_pretrained_full_train_pipeline
date: 2025-04-03
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`amazon_helpfulness_classification_on_tapt_pretrained_full_train_pipeline` is a English model originally trained by ltuzova.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/amazon_helpfulness_classification_on_tapt_pretrained_full_train_pipeline_en_5.5.1_3.0_1743694159259.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/amazon_helpfulness_classification_on_tapt_pretrained_full_train_pipeline_en_5.5.1_3.0_1743694159259.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("amazon_helpfulness_classification_on_tapt_pretrained_full_train_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("amazon_helpfulness_classification_on_tapt_pretrained_full_train_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|amazon_helpfulness_classification_on_tapt_pretrained_full_train_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|468.5 MB|

## References

https://huggingface.co/ltuzova/amazon_helpfulness_classification_on_TAPT_pretrained_full_train

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification