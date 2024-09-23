---
layout: model
title: English token_classification_test_sahithiankireddy_pipeline pipeline DistilBertForTokenClassification from sahithiankireddy
author: John Snow Labs
name: token_classification_test_sahithiankireddy_pipeline
date: 2024-09-08
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`token_classification_test_sahithiankireddy_pipeline` is a English model originally trained by sahithiankireddy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/token_classification_test_sahithiankireddy_pipeline_en_5.5.0_3.0_1725788965117.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/token_classification_test_sahithiankireddy_pipeline_en_5.5.0_3.0_1725788965117.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("token_classification_test_sahithiankireddy_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("token_classification_test_sahithiankireddy_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|token_classification_test_sahithiankireddy_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.4 MB|

## References

https://huggingface.co/sahithiankireddy/token_classification_test

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification