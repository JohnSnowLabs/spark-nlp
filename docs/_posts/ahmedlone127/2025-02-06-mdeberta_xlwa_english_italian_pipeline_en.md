---
layout: model
title: English mdeberta_xlwa_english_italian_pipeline pipeline DeBertaForQuestionAnswering from pgajo
author: John Snow Labs
name: mdeberta_xlwa_english_italian_pipeline
date: 2025-02-06
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained DeBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mdeberta_xlwa_english_italian_pipeline` is a English model originally trained by pgajo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mdeberta_xlwa_english_italian_pipeline_en_5.5.1_3.0_1738868814419.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mdeberta_xlwa_english_italian_pipeline_en_5.5.1_3.0_1738868814419.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mdeberta_xlwa_english_italian_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mdeberta_xlwa_english_italian_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mdeberta_xlwa_english_italian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|813.8 MB|

## References

https://huggingface.co/pgajo/mdeberta-xlwa-en-it

## Included Models

- MultiDocumentAssembler
- DeBertaForQuestionAnswering