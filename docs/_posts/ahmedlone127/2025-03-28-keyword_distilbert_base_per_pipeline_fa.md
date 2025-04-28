---
layout: model
title: Persian keyword_distilbert_base_per_pipeline pipeline DistilBertForTokenClassification from PakdamanAli
author: John Snow Labs
name: keyword_distilbert_base_per_pipeline
date: 2025-03-28
tags: [fa, open_source, pipeline, onnx]
task: Named Entity Recognition
language: fa
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`keyword_distilbert_base_per_pipeline` is a Persian model originally trained by PakdamanAli.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/keyword_distilbert_base_per_pipeline_fa_5.5.1_3.0_1743125130350.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/keyword_distilbert_base_per_pipeline_fa_5.5.1_3.0_1743125130350.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("keyword_distilbert_base_per_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("keyword_distilbert_base_per_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|keyword_distilbert_base_per_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|505.4 MB|

## References

https://huggingface.co/PakdamanAli/keyword_distilbert_base_per

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification