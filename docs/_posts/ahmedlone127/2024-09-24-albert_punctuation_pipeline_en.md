---
layout: model
title: English albert_punctuation_pipeline pipeline BertForTokenClassification from Wikidepia
author: John Snow Labs
name: albert_punctuation_pipeline
date: 2024-09-24
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albert_punctuation_pipeline` is a English model originally trained by Wikidepia.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_punctuation_pipeline_en_5.5.0_3.0_1727203077794.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_punctuation_pipeline_en_5.5.0_3.0_1727203077794.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("albert_punctuation_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("albert_punctuation_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_punctuation_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|42.0 MB|

## References

https://huggingface.co/Wikidepia/albert-punctuation

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification