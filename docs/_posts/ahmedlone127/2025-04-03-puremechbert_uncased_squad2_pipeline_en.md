---
layout: model
title: English puremechbert_uncased_squad2_pipeline pipeline BertForQuestionAnswering from CambridgeMolecularEngineering
author: John Snow Labs
name: puremechbert_uncased_squad2_pipeline
date: 2025-04-03
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`puremechbert_uncased_squad2_pipeline` is a English model originally trained by CambridgeMolecularEngineering.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/puremechbert_uncased_squad2_pipeline_en_5.5.1_3.0_1743648255409.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/puremechbert_uncased_squad2_pipeline_en_5.5.1_3.0_1743648255409.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("puremechbert_uncased_squad2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("puremechbert_uncased_squad2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|puremechbert_uncased_squad2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|400.0 MB|

## References

https://huggingface.co/CambridgeMolecularEngineering/PureMechBERT-uncased-squad2

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering