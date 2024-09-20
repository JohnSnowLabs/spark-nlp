---
layout: model
title: English code_segmentor_distilbert_pipeline pipeline DistilBertForTokenClassification from LemiSt
author: John Snow Labs
name: code_segmentor_distilbert_pipeline
date: 2024-09-01
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`code_segmentor_distilbert_pipeline` is a English model originally trained by LemiSt.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/code_segmentor_distilbert_pipeline_en_5.4.2_3.0_1725160746871.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/code_segmentor_distilbert_pipeline_en_5.4.2_3.0_1725160746871.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("code_segmentor_distilbert_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("code_segmentor_distilbert_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|code_segmentor_distilbert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|505.5 MB|

## References

https://huggingface.co/LemiSt/code-segmentor-distilbert

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification