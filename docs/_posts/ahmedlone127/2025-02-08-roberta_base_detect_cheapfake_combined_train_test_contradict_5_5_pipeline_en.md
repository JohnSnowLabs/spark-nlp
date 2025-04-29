---
layout: model
title: English roberta_base_detect_cheapfake_combined_train_test_contradict_5_5_pipeline pipeline RoBertaForSequenceClassification from hoanghoavienvo
author: John Snow Labs
name: roberta_base_detect_cheapfake_combined_train_test_contradict_5_5_pipeline
date: 2025-02-08
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_base_detect_cheapfake_combined_train_test_contradict_5_5_pipeline` is a English model originally trained by hoanghoavienvo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_detect_cheapfake_combined_train_test_contradict_5_5_pipeline_en_5.5.1_3.0_1739005238888.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_detect_cheapfake_combined_train_test_contradict_5_5_pipeline_en_5.5.1_3.0_1739005238888.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_base_detect_cheapfake_combined_train_test_contradict_5_5_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_base_detect_cheapfake_combined_train_test_contradict_5_5_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_detect_cheapfake_combined_train_test_contradict_5_5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|434.8 MB|

## References

https://huggingface.co/hoanghoavienvo/roberta-base-detect-cheapfake-combined-train-test-contradict-5-5

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification