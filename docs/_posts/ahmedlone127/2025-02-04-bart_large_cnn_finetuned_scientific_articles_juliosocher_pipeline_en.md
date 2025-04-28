---
layout: model
title: English bart_large_cnn_finetuned_scientific_articles_juliosocher_pipeline pipeline BartTransformer from juliosocher
author: John Snow Labs
name: bart_large_cnn_finetuned_scientific_articles_juliosocher_pipeline
date: 2025-02-04
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
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

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bart_large_cnn_finetuned_scientific_articles_juliosocher_pipeline` is a English model originally trained by juliosocher.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bart_large_cnn_finetuned_scientific_articles_juliosocher_pipeline_en_5.5.1_3.0_1738708249948.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bart_large_cnn_finetuned_scientific_articles_juliosocher_pipeline_en_5.5.1_3.0_1738708249948.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bart_large_cnn_finetuned_scientific_articles_juliosocher_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bart_large_cnn_finetuned_scientific_articles_juliosocher_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bart_large_cnn_finetuned_scientific_articles_juliosocher_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.9 GB|

## References

https://huggingface.co/juliosocher/bart-large-cnn-finetuned-scientific-articles

## Included Models

- DocumentAssembler
- BartTransformer