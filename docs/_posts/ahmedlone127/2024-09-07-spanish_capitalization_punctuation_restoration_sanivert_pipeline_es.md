---
layout: model
title: Castilian, Spanish spanish_capitalization_punctuation_restoration_sanivert_pipeline pipeline BertForTokenClassification from VOCALINLP
author: John Snow Labs
name: spanish_capitalization_punctuation_restoration_sanivert_pipeline
date: 2024-09-07
tags: [es, open_source, pipeline, onnx]
task: Named Entity Recognition
language: es
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`spanish_capitalization_punctuation_restoration_sanivert_pipeline` is a Castilian, Spanish model originally trained by VOCALINLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spanish_capitalization_punctuation_restoration_sanivert_pipeline_es_5.5.0_3.0_1725726630748.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/spanish_capitalization_punctuation_restoration_sanivert_pipeline_es_5.5.0_3.0_1725726630748.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("spanish_capitalization_punctuation_restoration_sanivert_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("spanish_capitalization_punctuation_restoration_sanivert_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|spanish_capitalization_punctuation_restoration_sanivert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|409.8 MB|

## References

https://huggingface.co/VOCALINLP/spanish_capitalization_punctuation_restoration_sanivert

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification