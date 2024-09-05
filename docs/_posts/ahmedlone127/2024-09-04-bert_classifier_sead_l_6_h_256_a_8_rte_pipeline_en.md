---
layout: model
title: English bert_classifier_sead_l_6_h_256_a_8_rte_pipeline pipeline BertForSequenceClassification from course5i
author: John Snow Labs
name: bert_classifier_sead_l_6_h_256_a_8_rte_pipeline
date: 2024-09-04
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_classifier_sead_l_6_h_256_a_8_rte_pipeline` is a English model originally trained by course5i.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_sead_l_6_h_256_a_8_rte_pipeline_en_5.5.0_3.0_1725432612233.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_sead_l_6_h_256_a_8_rte_pipeline_en_5.5.0_3.0_1725432612233.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_classifier_sead_l_6_h_256_a_8_rte_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_classifier_sead_l_6_h_256_a_8_rte_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_sead_l_6_h_256_a_8_rte_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|47.3 MB|

## References

https://huggingface.co/course5i/SEAD-L-6_H-256_A-8-rte

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification