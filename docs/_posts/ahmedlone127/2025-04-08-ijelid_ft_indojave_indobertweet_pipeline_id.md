---
layout: model
title: Indonesian ijelid_ft_indojave_indobertweet_pipeline pipeline BertForTokenClassification from fathan
author: John Snow Labs
name: ijelid_ft_indojave_indobertweet_pipeline
date: 2025-04-08
tags: [id, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ijelid_ft_indojave_indobertweet_pipeline` is a Indonesian model originally trained by fathan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ijelid_ft_indojave_indobertweet_pipeline_id_5.5.1_3.0_1744131943714.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ijelid_ft_indojave_indobertweet_pipeline_id_5.5.1_3.0_1744131943714.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ijelid_ft_indojave_indobertweet_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ijelid_ft_indojave_indobertweet_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ijelid_ft_indojave_indobertweet_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|411.9 MB|

## References

https://huggingface.co/fathan/ijelid-ft-indojave-indobertweet

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification