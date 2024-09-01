---
layout: model
title: French qamembert_large_pipeline pipeline CamemBertForQuestionAnswering from CATIE-AQ
author: John Snow Labs
name: qamembert_large_pipeline
date: 2024-09-01
tags: [fr, open_source, pipeline, onnx]
task: Question Answering
language: fr
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`qamembert_large_pipeline` is a French model originally trained by CATIE-AQ.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/qamembert_large_pipeline_fr_5.4.2_3.0_1725162175947.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/qamembert_large_pipeline_fr_5.4.2_3.0_1725162175947.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("qamembert_large_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("qamembert_large_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|qamembert_large_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|1.3 GB|

## References

https://huggingface.co/CATIE-AQ/QAmembert-large

## Included Models

- MultiDocumentAssembler
- CamemBertForQuestionAnswering