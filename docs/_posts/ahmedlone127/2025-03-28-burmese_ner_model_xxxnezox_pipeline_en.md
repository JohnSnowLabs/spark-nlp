---
layout: model
title: English burmese_ner_model_xxxnezox_pipeline pipeline DistilBertForTokenClassification from xxxNezox
author: John Snow Labs
name: burmese_ner_model_xxxnezox_pipeline
date: 2025-03-28
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`burmese_ner_model_xxxnezox_pipeline` is a English model originally trained by xxxNezox.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/burmese_ner_model_xxxnezox_pipeline_en_5.5.1_3.0_1743124900404.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/burmese_ner_model_xxxnezox_pipeline_en_5.5.1_3.0_1743124900404.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("burmese_ner_model_xxxnezox_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("burmese_ner_model_xxxnezox_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|burmese_ner_model_xxxnezox_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.3 MB|

## References

https://huggingface.co/xxxNezox/my_ner_model

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification