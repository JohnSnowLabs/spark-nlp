---
layout: model
title: Hungarian hunembert3_pipeline pipeline BertForSequenceClassification from poltextlab
author: John Snow Labs
name: hunembert3_pipeline
date: 2024-09-13
tags: [hu, open_source, pipeline, onnx]
task: Text Classification
language: hu
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hunembert3_pipeline` is a Hungarian model originally trained by poltextlab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hunembert3_pipeline_hu_5.5.0_3.0_1726201703167.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hunembert3_pipeline_hu_5.5.0_3.0_1726201703167.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hunembert3_pipeline", lang = "hu")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hunembert3_pipeline", lang = "hu")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hunembert3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|hu|
|Size:|414.7 MB|

## References

https://huggingface.co/poltextlab/HunEmBERT3

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification