---
layout: model
title: French section_classifier_pipeline pipeline BertForSequenceClassification from TakedaAIML
author: John Snow Labs
name: section_classifier_pipeline
date: 2025-04-06
tags: [fr, open_source, pipeline, onnx]
task: Text Classification
language: fr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`section_classifier_pipeline` is a French model originally trained by TakedaAIML.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/section_classifier_pipeline_fr_5.5.1_3.0_1743962122681.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/section_classifier_pipeline_fr_5.5.1_3.0_1743962122681.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("section_classifier_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("section_classifier_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|section_classifier_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|627.8 MB|

## References

https://huggingface.co/TakedaAIML/section_classifier

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification