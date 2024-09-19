---
layout: model
title: Hindi readabert_hindi_pipeline pipeline BertForSequenceClassification from tareknaous
author: John Snow Labs
name: readabert_hindi_pipeline
date: 2024-09-18
tags: [hi, open_source, pipeline, onnx]
task: Text Classification
language: hi
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`readabert_hindi_pipeline` is a Hindi model originally trained by tareknaous.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/readabert_hindi_pipeline_hi_5.5.0_3.0_1726647407029.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/readabert_hindi_pipeline_hi_5.5.0_3.0_1726647407029.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("readabert_hindi_pipeline", lang = "hi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("readabert_hindi_pipeline", lang = "hi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|readabert_hindi_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|892.7 MB|

## References

https://huggingface.co/tareknaous/readabert-hi

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification