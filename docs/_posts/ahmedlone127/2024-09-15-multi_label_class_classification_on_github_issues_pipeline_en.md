---
layout: model
title: English multi_label_class_classification_on_github_issues_pipeline pipeline BertForSequenceClassification from Rami
author: John Snow Labs
name: multi_label_class_classification_on_github_issues_pipeline
date: 2024-09-15
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multi_label_class_classification_on_github_issues_pipeline` is a English model originally trained by Rami.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multi_label_class_classification_on_github_issues_pipeline_en_5.5.0_3.0_1726375923874.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multi_label_class_classification_on_github_issues_pipeline_en_5.5.0_3.0_1726375923874.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("multi_label_class_classification_on_github_issues_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("multi_label_class_classification_on_github_issues_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multi_label_class_classification_on_github_issues_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|411.5 MB|

## References

https://huggingface.co/Rami/multi-label-class-classification-on-github-issues

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification