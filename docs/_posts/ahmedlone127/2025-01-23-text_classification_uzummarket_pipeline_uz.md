---
layout: model
title: Uzbek text_classification_uzummarket_pipeline pipeline XlmRoBertaForSequenceClassification from fanaf91318
author: John Snow Labs
name: text_classification_uzummarket_pipeline
date: 2025-01-23
tags: [uz, open_source, pipeline, onnx]
task: Text Classification
language: uz
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`text_classification_uzummarket_pipeline` is a Uzbek model originally trained by fanaf91318.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/text_classification_uzummarket_pipeline_uz_5.5.1_3.0_1737653954503.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/text_classification_uzummarket_pipeline_uz_5.5.1_3.0_1737653954503.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("text_classification_uzummarket_pipeline", lang = "uz")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("text_classification_uzummarket_pipeline", lang = "uz")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|text_classification_uzummarket_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|uz|
|Size:|825.5 MB|

## References

https://huggingface.co/fanaf91318/text-classification-uzummarket

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification