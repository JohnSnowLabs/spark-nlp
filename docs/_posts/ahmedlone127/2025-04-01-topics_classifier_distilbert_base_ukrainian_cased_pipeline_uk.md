---
layout: model
title: Ukrainian topics_classifier_distilbert_base_ukrainian_cased_pipeline pipeline DistilBertForSequenceClassification from ua-l
author: John Snow Labs
name: topics_classifier_distilbert_base_ukrainian_cased_pipeline
date: 2025-04-01
tags: [uk, open_source, pipeline, onnx]
task: Text Classification
language: uk
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`topics_classifier_distilbert_base_ukrainian_cased_pipeline` is a Ukrainian model originally trained by ua-l.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/topics_classifier_distilbert_base_ukrainian_cased_pipeline_uk_5.5.1_3.0_1743538520848.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/topics_classifier_distilbert_base_ukrainian_cased_pipeline_uk_5.5.1_3.0_1743538520848.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("topics_classifier_distilbert_base_ukrainian_cased_pipeline", lang = "uk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("topics_classifier_distilbert_base_ukrainian_cased_pipeline", lang = "uk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|topics_classifier_distilbert_base_ukrainian_cased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|uk|
|Size:|197.2 MB|

## References

https://huggingface.co/ua-l/topics-classifier-distilbert-base-uk-cased

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification