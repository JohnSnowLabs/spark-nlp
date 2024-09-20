---
layout: model
title: French esg_classification_french_english_pipeline pipeline DistilBertForSequenceClassification from cea-list-lasti
author: John Snow Labs
name: esg_classification_french_english_pipeline
date: 2024-09-05
tags: [fr, open_source, pipeline, onnx]
task: Text Classification
language: fr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`esg_classification_french_english_pipeline` is a French model originally trained by cea-list-lasti.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/esg_classification_french_english_pipeline_fr_5.5.0_3.0_1725506980005.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/esg_classification_french_english_pipeline_fr_5.5.0_3.0_1725506980005.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("esg_classification_french_english_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("esg_classification_french_english_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|esg_classification_french_english_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|508.0 MB|

## References

https://huggingface.co/cea-list-lasti/ESG-classification-fr-en

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification