---
layout: model
title: English sentence_similarity_finetuned_mpnet_adrta_pipeline pipeline MPNetForSequenceClassification from aizenSosuke
author: John Snow Labs
name: sentence_similarity_finetuned_mpnet_adrta_pipeline
date: 2024-11-11
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

Pretrained MPNetForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sentence_similarity_finetuned_mpnet_adrta_pipeline` is a English model originally trained by aizenSosuke.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_similarity_finetuned_mpnet_adrta_pipeline_en_5.5.1_3.0_1731301620394.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_similarity_finetuned_mpnet_adrta_pipeline_en_5.5.1_3.0_1731301620394.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sentence_similarity_finetuned_mpnet_adrta_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sentence_similarity_finetuned_mpnet_adrta_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentence_similarity_finetuned_mpnet_adrta_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.3 MB|

## References

https://huggingface.co/aizenSosuke/sentence-similarity-finetuned-mpnet-adrta

## Included Models

- DocumentAssembler
- TokenizerModel
- MPNetForSequenceClassification