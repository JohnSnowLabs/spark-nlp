---
layout: model
title: Tamil tamil_bert_l3_cube_relative_position_pipeline pipeline BertForSequenceClassification from jusgowiturs
author: John Snow Labs
name: tamil_bert_l3_cube_relative_position_pipeline
date: 2025-04-09
tags: [ta, open_source, pipeline, onnx]
task: Text Classification
language: ta
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tamil_bert_l3_cube_relative_position_pipeline` is a Tamil model originally trained by jusgowiturs.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tamil_bert_l3_cube_relative_position_pipeline_ta_5.5.1_3.0_1744180321786.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tamil_bert_l3_cube_relative_position_pipeline_ta_5.5.1_3.0_1744180321786.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tamil_bert_l3_cube_relative_position_pipeline", lang = "ta")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tamil_bert_l3_cube_relative_position_pipeline", lang = "ta")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tamil_bert_l3_cube_relative_position_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ta|
|Size:|894.4 MB|

## References

https://huggingface.co/jusgowiturs/Tamil_Bert_L3_Cube_Relative_Position

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification