---
layout: model
title: English roberta_base_mixed_vanilla_time_reversed_pipeline pipeline RoBertaForSequenceClassification from research-dump
author: John Snow Labs
name: roberta_base_mixed_vanilla_time_reversed_pipeline
date: 2025-02-07
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_base_mixed_vanilla_time_reversed_pipeline` is a English model originally trained by research-dump.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_mixed_vanilla_time_reversed_pipeline_en_5.5.1_3.0_1738921439986.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_mixed_vanilla_time_reversed_pipeline_en_5.5.1_3.0_1738921439986.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_base_mixed_vanilla_time_reversed_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_base_mixed_vanilla_time_reversed_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_mixed_vanilla_time_reversed_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|431.4 MB|

## References

https://huggingface.co/research-dump/roberta-base_mixed_vanilla_time_reversed

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification