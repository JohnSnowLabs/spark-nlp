---
layout: model
title: Portuguese perovaz_portuguese_breton_classifier_pipeline pipeline BertForSequenceClassification from Bastao
author: John Snow Labs
name: perovaz_portuguese_breton_classifier_pipeline
date: 2025-04-03
tags: [pt, open_source, pipeline, onnx]
task: Text Classification
language: pt
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`perovaz_portuguese_breton_classifier_pipeline` is a Portuguese model originally trained by Bastao.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/perovaz_portuguese_breton_classifier_pipeline_pt_5.5.1_3.0_1743719462854.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/perovaz_portuguese_breton_classifier_pipeline_pt_5.5.1_3.0_1743719462854.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("perovaz_portuguese_breton_classifier_pipeline", lang = "pt")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("perovaz_portuguese_breton_classifier_pipeline", lang = "pt")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|perovaz_portuguese_breton_classifier_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|16.8 MB|

## References

References

https://huggingface.co/Bastao/PeroVaz_PT-BR_Classifier

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification