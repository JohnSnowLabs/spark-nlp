---
layout: model
title: English deberta_base_us_pipeline pipeline DeBertaForSequenceClassification from Cartinoe5930
author: John Snow Labs
name: deberta_base_us_pipeline
date: 2025-02-03
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`deberta_base_us_pipeline` is a English model originally trained by Cartinoe5930.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_base_us_pipeline_en_5.5.1_3.0_1738618777872.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_base_us_pipeline_en_5.5.1_3.0_1738618777872.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("deberta_base_us_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("deberta_base_us_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_base_us_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|604.3 MB|

## References

https://huggingface.co/Cartinoe5930/deberta-base-us

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification