---
layout: model
title: Multilingual cat_1_html_distilbert_base_multilingual_cased_v2_pipeline pipeline DistilBertForSequenceClassification from chuuhtetnaing
author: John Snow Labs
name: cat_1_html_distilbert_base_multilingual_cased_v2_pipeline
date: 2025-02-08
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cat_1_html_distilbert_base_multilingual_cased_v2_pipeline` is a Multilingual model originally trained by chuuhtetnaing.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cat_1_html_distilbert_base_multilingual_cased_v2_pipeline_xx_5.5.1_3.0_1739010744369.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cat_1_html_distilbert_base_multilingual_cased_v2_pipeline_xx_5.5.1_3.0_1739010744369.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cat_1_html_distilbert_base_multilingual_cased_v2_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cat_1_html_distilbert_base_multilingual_cased_v2_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cat_1_html_distilbert_base_multilingual_cased_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|507.6 MB|

## References

https://huggingface.co/chuuhtetnaing/cat-1-html-distilbert-base-multilingual-cased-v2

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification