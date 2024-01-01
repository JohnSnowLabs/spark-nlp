---
layout: model
title: English caption_title_pipeline pipeline DistilBertForSequenceClassification from tgieruc
author: John Snow Labs
name: caption_title_pipeline
date: 2023-12-29
tags: [bert, en, open_source, sequence_classification, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.2.2
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`caption_title_pipeline` is a English model originally trained by tgieruc.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/caption_title_pipeline_en_5.2.2_3.0_1703867727027.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/caption_title_pipeline_en_5.2.2_3.0_1703867727027.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline('caption_title_pipeline', lang = 'en')
annotations =  pipeline.transform(df)

```
```scala

val pipeline = new PretrainedPipeline('caption_title_pipeline', lang = 'en')
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|caption_title_pipeline|
|Compatibility:|Spark NLP 5.2.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/tgieruc/caption-title-pipeline