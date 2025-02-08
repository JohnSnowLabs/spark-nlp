---
layout: model
title: Arabic sent_arabic_quran_nahj_sahife_pipeline pipeline BertSentenceEmbeddings from pourmand1376
author: John Snow Labs
name: sent_arabic_quran_nahj_sahife_pipeline
date: 2025-02-06
tags: [ar, open_source, pipeline, onnx]
task: Embeddings
language: ar
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_arabic_quran_nahj_sahife_pipeline` is a Arabic model originally trained by pourmand1376.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_arabic_quran_nahj_sahife_pipeline_ar_5.5.1_3.0_1738832022058.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_arabic_quran_nahj_sahife_pipeline_ar_5.5.1_3.0_1738832022058.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_arabic_quran_nahj_sahife_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_arabic_quran_nahj_sahife_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_arabic_quran_nahj_sahife_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|412.5 MB|

## References

https://huggingface.co/pourmand1376/arabic-quran-nahj-sahife

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings