---
layout: model
title: English debertabaseemotionbalanced_pipeline pipeline DeBertaForSequenceClassification from aliciiavs
author: John Snow Labs
name: debertabaseemotionbalanced_pipeline
date: 2024-09-04
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`debertabaseemotionbalanced_pipeline` is a English model originally trained by aliciiavs.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/debertabaseemotionbalanced_pipeline_en_5.5.0_3.0_1725469360475.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/debertabaseemotionbalanced_pipeline_en_5.5.0_3.0_1725469360475.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("debertabaseemotionbalanced_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("debertabaseemotionbalanced_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|debertabaseemotionbalanced_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|593.4 MB|

## References

https://huggingface.co/aliciiavs/debertabaseemotionbalanced

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification