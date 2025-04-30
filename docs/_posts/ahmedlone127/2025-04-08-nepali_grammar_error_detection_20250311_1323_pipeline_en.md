---
layout: model
title: English nepali_grammar_error_detection_20250311_1323_pipeline pipeline BertForSequenceClassification from rishikeshgautam
author: John Snow Labs
name: nepali_grammar_error_detection_20250311_1323_pipeline
date: 2025-04-08
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nepali_grammar_error_detection_20250311_1323_pipeline` is a English model originally trained by rishikeshgautam.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nepali_grammar_error_detection_20250311_1323_pipeline_en_5.5.1_3.0_1744116103688.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nepali_grammar_error_detection_20250311_1323_pipeline_en_5.5.1_3.0_1744116103688.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nepali_grammar_error_detection_20250311_1323_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nepali_grammar_error_detection_20250311_1323_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nepali_grammar_error_detection_20250311_1323_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|892.7 MB|

## References

https://huggingface.co/rishikeshgautam/nepali-grammar-error-detection-20250311_1323

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification