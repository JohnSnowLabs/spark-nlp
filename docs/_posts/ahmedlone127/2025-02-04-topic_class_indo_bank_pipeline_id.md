---
layout: model
title: Indonesian topic_class_indo_bank_pipeline pipeline RoBertaForSequenceClassification from dhanikitkat
author: John Snow Labs
name: topic_class_indo_bank_pipeline
date: 2025-02-04
tags: [id, open_source, pipeline, onnx]
task: Text Classification
language: id
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`topic_class_indo_bank_pipeline` is a Indonesian model originally trained by dhanikitkat.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/topic_class_indo_bank_pipeline_id_5.5.1_3.0_1738691259197.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/topic_class_indo_bank_pipeline_id_5.5.1_3.0_1738691259197.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("topic_class_indo_bank_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("topic_class_indo_bank_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|topic_class_indo_bank_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|308.4 MB|

## References

https://huggingface.co/dhanikitkat/topic-class-indo-bank

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification