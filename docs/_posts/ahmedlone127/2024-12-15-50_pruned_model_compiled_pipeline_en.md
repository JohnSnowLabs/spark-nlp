---
layout: model
title: English 50_pruned_model_compiled_pipeline pipeline BertForQuestionAnswering from xihajun
author: John Snow Labs
name: 50_pruned_model_compiled_pipeline
date: 2024-12-15
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`50_pruned_model_compiled_pipeline` is a English model originally trained by xihajun.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/50_pruned_model_compiled_pipeline_en_5.5.1_3.0_1734222358897.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/50_pruned_model_compiled_pipeline_en_5.5.1_3.0_1734222358897.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("50_pruned_model_compiled_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("50_pruned_model_compiled_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|50_pruned_model_compiled_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|781.5 MB|

## References

https://huggingface.co/xihajun/50_pruned_model_compiled

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering