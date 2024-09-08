---
layout: model
title: English dynamic_tinybert_finetuned_squad_pipeline pipeline BertForQuestionAnswering from phdev
author: John Snow Labs
name: dynamic_tinybert_finetuned_squad_pipeline
date: 2024-09-05
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`dynamic_tinybert_finetuned_squad_pipeline` is a English model originally trained by phdev.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dynamic_tinybert_finetuned_squad_pipeline_en_5.5.0_3.0_1725560277695.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dynamic_tinybert_finetuned_squad_pipeline_en_5.5.0_3.0_1725560277695.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("dynamic_tinybert_finetuned_squad_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("dynamic_tinybert_finetuned_squad_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dynamic_tinybert_finetuned_squad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.0 MB|

## References

https://huggingface.co/phdev/dynamic_tinybert-finetuned-squad

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering