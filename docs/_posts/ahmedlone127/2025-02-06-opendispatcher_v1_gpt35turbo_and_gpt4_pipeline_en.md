---
layout: model
title: English opendispatcher_v1_gpt35turbo_and_gpt4_pipeline pipeline DistilBertForSequenceClassification from gaodrew
author: John Snow Labs
name: opendispatcher_v1_gpt35turbo_and_gpt4_pipeline
date: 2025-02-06
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`opendispatcher_v1_gpt35turbo_and_gpt4_pipeline` is a English model originally trained by gaodrew.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/opendispatcher_v1_gpt35turbo_and_gpt4_pipeline_en_5.5.1_3.0_1738812604254.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/opendispatcher_v1_gpt35turbo_and_gpt4_pipeline_en_5.5.1_3.0_1738812604254.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("opendispatcher_v1_gpt35turbo_and_gpt4_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("opendispatcher_v1_gpt35turbo_and_gpt4_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|opendispatcher_v1_gpt35turbo_and_gpt4_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/gaodrew/OpenDispatcher_v1_gpt35turbo_and_gpt4

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification