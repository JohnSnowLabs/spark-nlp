---
layout: model
title: English esgbertv1_access_tonga_tonga_islands_communications_pipeline pipeline BertForSequenceClassification from ab3223323
author: John Snow Labs
name: esgbertv1_access_tonga_tonga_islands_communications_pipeline
date: 2025-01-27
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`esgbertv1_access_tonga_tonga_islands_communications_pipeline` is a English model originally trained by ab3223323.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/esgbertv1_access_tonga_tonga_islands_communications_pipeline_en_5.5.1_3.0_1738005383177.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/esgbertv1_access_tonga_tonga_islands_communications_pipeline_en_5.5.1_3.0_1738005383177.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("esgbertv1_access_tonga_tonga_islands_communications_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("esgbertv1_access_tonga_tonga_islands_communications_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|esgbertv1_access_tonga_tonga_islands_communications_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.4 MB|

## References

https://huggingface.co/ab3223323/esgBERTv1_Access_to_Communications

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification