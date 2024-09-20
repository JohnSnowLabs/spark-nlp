---
layout: model
title: English distilbert_persian_farsi_zwnj_base_finetuned_lm_pipeline pipeline DistilBertEmbeddings from 4h0r4
author: John Snow Labs
name: distilbert_persian_farsi_zwnj_base_finetuned_lm_pipeline
date: 2024-09-03
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained DistilBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_persian_farsi_zwnj_base_finetuned_lm_pipeline` is a English model originally trained by 4h0r4.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_persian_farsi_zwnj_base_finetuned_lm_pipeline_en_5.5.0_3.0_1725389504137.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_persian_farsi_zwnj_base_finetuned_lm_pipeline_en_5.5.0_3.0_1725389504137.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilbert_persian_farsi_zwnj_base_finetuned_lm_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilbert_persian_farsi_zwnj_base_finetuned_lm_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_persian_farsi_zwnj_base_finetuned_lm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|282.3 MB|

## References

https://huggingface.co/4h0r4/distilbert-fa-zwnj-base-finetuned-lm

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertEmbeddings