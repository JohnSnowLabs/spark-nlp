---
layout: model
title: English robertacrawlpt_base_news_pipeline pipeline RoBertaForSequenceClassification from Angelo-Magno
author: John Snow Labs
name: robertacrawlpt_base_news_pipeline
date: 2025-02-05
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`robertacrawlpt_base_news_pipeline` is a English model originally trained by Angelo-Magno.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/robertacrawlpt_base_news_pipeline_en_5.5.1_3.0_1738799347710.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/robertacrawlpt_base_news_pipeline_en_5.5.1_3.0_1738799347710.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("robertacrawlpt_base_news_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("robertacrawlpt_base_news_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|robertacrawlpt_base_news_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|448.5 MB|

## References

https://huggingface.co/Angelo-Magno/RoBERTaCrawlPT-base-news

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification