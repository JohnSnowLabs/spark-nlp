---
layout: model
title: English by_the_horns_t45gysbert_pipeline pipeline BertForTokenClassification from ArjanvD95
author: John Snow Labs
name: by_the_horns_t45gysbert_pipeline
date: 2025-04-04
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`by_the_horns_t45gysbert_pipeline` is a English model originally trained by ArjanvD95.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/by_the_horns_t45gysbert_pipeline_en_5.5.1_3.0_1743743254100.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/by_the_horns_t45gysbert_pipeline_en_5.5.1_3.0_1743743254100.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("by_the_horns_t45gysbert_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("by_the_horns_t45gysbert_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|by_the_horns_t45gysbert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|408.2 MB|

## References

https://huggingface.co/ArjanvD95/by_the_horns_T45Gysbert

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification