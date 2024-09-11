---
layout: model
title: English sciverbinary_model_train_dev_data_robertal_label_neutral_detector_pipeline pipeline RoBertaForSequenceClassification from xiazeng
author: John Snow Labs
name: sciverbinary_model_train_dev_data_robertal_label_neutral_detector_pipeline
date: 2024-09-10
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sciverbinary_model_train_dev_data_robertal_label_neutral_detector_pipeline` is a English model originally trained by xiazeng.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sciverbinary_model_train_dev_data_robertal_label_neutral_detector_pipeline_en_5.5.0_3.0_1725966153604.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sciverbinary_model_train_dev_data_robertal_label_neutral_detector_pipeline_en_5.5.0_3.0_1725966153604.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sciverbinary_model_train_dev_data_robertal_label_neutral_detector_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sciverbinary_model_train_dev_data_robertal_label_neutral_detector_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sciverbinary_model_train_dev_data_robertal_label_neutral_detector_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/xiazeng/sciverbinary-model_train_dev_data-robertal-label-neutral_detector

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification