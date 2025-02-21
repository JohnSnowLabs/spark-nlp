---
layout: model
title: English judge_answer__22_deberta_v3_large_fresh_repo_pipeline pipeline DeBertaForSequenceClassification from tom-010
author: John Snow Labs
name: judge_answer__22_deberta_v3_large_fresh_repo_pipeline
date: 2025-01-24
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`judge_answer__22_deberta_v3_large_fresh_repo_pipeline` is a English model originally trained by tom-010.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/judge_answer__22_deberta_v3_large_fresh_repo_pipeline_en_5.5.1_3.0_1737728264386.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/judge_answer__22_deberta_v3_large_fresh_repo_pipeline_en_5.5.1_3.0_1737728264386.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("judge_answer__22_deberta_v3_large_fresh_repo_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("judge_answer__22_deberta_v3_large_fresh_repo_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|judge_answer__22_deberta_v3_large_fresh_repo_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.6 GB|

## References

https://huggingface.co/tom-010/judge_answer__22_deberta_v3_large_fresh_repo

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification