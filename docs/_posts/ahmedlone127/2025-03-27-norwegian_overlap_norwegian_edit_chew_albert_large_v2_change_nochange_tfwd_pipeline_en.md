---
layout: model
title: English norwegian_overlap_norwegian_edit_chew_albert_large_v2_change_nochange_tfwd_pipeline pipeline AlbertForSequenceClassification from research-dump
author: John Snow Labs
name: norwegian_overlap_norwegian_edit_chew_albert_large_v2_change_nochange_tfwd_pipeline
date: 2025-03-27
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

Pretrained AlbertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`norwegian_overlap_norwegian_edit_chew_albert_large_v2_change_nochange_tfwd_pipeline` is a English model originally trained by research-dump.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/norwegian_overlap_norwegian_edit_chew_albert_large_v2_change_nochange_tfwd_pipeline_en_5.5.1_3.0_1743073773335.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/norwegian_overlap_norwegian_edit_chew_albert_large_v2_change_nochange_tfwd_pipeline_en_5.5.1_3.0_1743073773335.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("norwegian_overlap_norwegian_edit_chew_albert_large_v2_change_nochange_tfwd_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("norwegian_overlap_norwegian_edit_chew_albert_large_v2_change_nochange_tfwd_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|norwegian_overlap_norwegian_edit_chew_albert_large_v2_change_nochange_tfwd_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|66.7 MB|

## References

https://huggingface.co/research-dump/no_overlap_no_edit_chew_albert-large-v2_change_nochange_tfwd

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertForSequenceClassification