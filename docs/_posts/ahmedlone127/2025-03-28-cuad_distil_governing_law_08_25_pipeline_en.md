---
layout: model
title: English cuad_distil_governing_law_08_25_pipeline pipeline DistilBertForQuestionAnswering from saraks
author: John Snow Labs
name: cuad_distil_governing_law_08_25_pipeline
date: 2025-03-28
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

Pretrained DistilBertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cuad_distil_governing_law_08_25_pipeline` is a English model originally trained by saraks.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cuad_distil_governing_law_08_25_pipeline_en_5.5.1_3.0_1743125990857.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cuad_distil_governing_law_08_25_pipeline_en_5.5.1_3.0_1743125990857.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cuad_distil_governing_law_08_25_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cuad_distil_governing_law_08_25_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cuad_distil_governing_law_08_25_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.3 MB|

## References

https://huggingface.co/saraks/cuad-distil-governing_law-08-25

## Included Models

- MultiDocumentAssembler
- DistilBertForQuestionAnswering