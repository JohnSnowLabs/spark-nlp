---
layout: model
title: Russian mil_class_exp_sber_balanssedclass_pipeline pipeline BertForSequenceClassification from bodomerka
author: John Snow Labs
name: mil_class_exp_sber_balanssedclass_pipeline
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mil_class_exp_sber_balanssedclass_pipeline` is a Russian model originally trained by bodomerka.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mil_class_exp_sber_balanssedclass_pipeline_ru_5.5.1_3.0_1738591779607.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mil_class_exp_sber_balanssedclass_pipeline_ru_5.5.1_3.0_1738591779607.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mil_class_exp_sber_balanssedclass_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mil_class_exp_sber_balanssedclass_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mil_class_exp_sber_balanssedclass_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|669.3 MB|

## References

https://huggingface.co/bodomerka/Mil_class_exp_sber_balanssedclass

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification