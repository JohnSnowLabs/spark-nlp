---
layout: model
title: Russian milytary_exp_class_classification_sber_ai_based_pipeline pipeline BertForSequenceClassification from bodomerka
author: John Snow Labs
name: milytary_exp_class_classification_sber_ai_based_pipeline
date: 2025-02-03
tags: [ru, open_source, pipeline, onnx]
task: Text Classification
language: ru
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`milytary_exp_class_classification_sber_ai_based_pipeline` is a Russian model originally trained by bodomerka.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/milytary_exp_class_classification_sber_ai_based_pipeline_ru_5.5.1_3.0_1738541730204.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/milytary_exp_class_classification_sber_ai_based_pipeline_ru_5.5.1_3.0_1738541730204.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("milytary_exp_class_classification_sber_ai_based_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("milytary_exp_class_classification_sber_ai_based_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|milytary_exp_class_classification_sber_ai_based_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|669.3 MB|

## References

https://huggingface.co/bodomerka/Milytary_exp_class_classification_sber_ai_based

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification