---
layout: model
title: English distilbertfinetunehs5e8bhlr_karthikalagarsamy_pipeline pipeline DistilBertForQuestionAnswering from KarthikAlagarsamy
author: John Snow Labs
name: distilbertfinetunehs5e8bhlr_karthikalagarsamy_pipeline
date: 2024-12-16
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

Pretrained DistilBertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbertfinetunehs5e8bhlr_karthikalagarsamy_pipeline` is a English model originally trained by KarthikAlagarsamy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbertfinetunehs5e8bhlr_karthikalagarsamy_pipeline_en_5.5.1_3.0_1734348279146.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbertfinetunehs5e8bhlr_karthikalagarsamy_pipeline_en_5.5.1_3.0_1734348279146.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilbertfinetunehs5e8bhlr_karthikalagarsamy_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilbertfinetunehs5e8bhlr_karthikalagarsamy_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbertfinetunehs5e8bhlr_karthikalagarsamy_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.2 MB|

## References

https://huggingface.co/KarthikAlagarsamy/distilbertfinetuneHS5E8BHLR

## Included Models

- MultiDocumentAssembler
- DistilBertForQuestionAnswering