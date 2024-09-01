---
layout: model
title: English airbnb_reviews_helpfulness_classifier_roberta_base_pipeline pipeline RoBertaForSequenceClassification from lihuicham
author: John Snow Labs
name: airbnb_reviews_helpfulness_classifier_roberta_base_pipeline
date: 2024-09-01
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`airbnb_reviews_helpfulness_classifier_roberta_base_pipeline` is a English model originally trained by lihuicham.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/airbnb_reviews_helpfulness_classifier_roberta_base_pipeline_en_5.4.2_3.0_1725194740534.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/airbnb_reviews_helpfulness_classifier_roberta_base_pipeline_en_5.4.2_3.0_1725194740534.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("airbnb_reviews_helpfulness_classifier_roberta_base_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("airbnb_reviews_helpfulness_classifier_roberta_base_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|airbnb_reviews_helpfulness_classifier_roberta_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|427.9 MB|

## References

https://huggingface.co/lihuicham/airbnb-reviews-helpfulness-classifier-roberta-base

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification