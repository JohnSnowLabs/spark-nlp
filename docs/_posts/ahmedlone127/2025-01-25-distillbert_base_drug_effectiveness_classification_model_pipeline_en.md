---
layout: model
title: English distillbert_base_drug_effectiveness_classification_model_pipeline pipeline DistilBertForSequenceClassification from jvbjkbjkbfjis
author: John Snow Labs
name: distillbert_base_drug_effectiveness_classification_model_pipeline
date: 2025-01-25
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distillbert_base_drug_effectiveness_classification_model_pipeline` is a English model originally trained by jvbjkbjkbfjis.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distillbert_base_drug_effectiveness_classification_model_pipeline_en_5.5.1_3.0_1737822075450.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distillbert_base_drug_effectiveness_classification_model_pipeline_en_5.5.1_3.0_1737822075450.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distillbert_base_drug_effectiveness_classification_model_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distillbert_base_drug_effectiveness_classification_model_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distillbert_base_drug_effectiveness_classification_model_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/jvbjkbjkbfjis/distillbert-base-drug-effectiveness-classification-model

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification