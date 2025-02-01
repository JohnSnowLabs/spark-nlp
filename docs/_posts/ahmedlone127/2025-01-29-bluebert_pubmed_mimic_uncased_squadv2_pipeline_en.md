---
layout: model
title: English bluebert_pubmed_mimic_uncased_squadv2_pipeline pipeline BertForQuestionAnswering from trevorkwan
author: John Snow Labs
name: bluebert_pubmed_mimic_uncased_squadv2_pipeline
date: 2025-01-29
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bluebert_pubmed_mimic_uncased_squadv2_pipeline` is a English model originally trained by trevorkwan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bluebert_pubmed_mimic_uncased_squadv2_pipeline_en_5.5.1_3.0_1738186040206.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bluebert_pubmed_mimic_uncased_squadv2_pipeline_en_5.5.1_3.0_1738186040206.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bluebert_pubmed_mimic_uncased_squadv2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bluebert_pubmed_mimic_uncased_squadv2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bluebert_pubmed_mimic_uncased_squadv2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.1 MB|

## References

https://huggingface.co/trevorkwan/bluebert_pubmed_mimic_uncased_squadv2

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering