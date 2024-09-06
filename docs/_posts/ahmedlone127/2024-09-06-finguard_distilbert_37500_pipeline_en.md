---
layout: model
title: English finguard_distilbert_37500_pipeline pipeline DistilBertForTokenClassification from AnirudhLanka2002
author: John Snow Labs
name: finguard_distilbert_37500_pipeline
date: 2024-09-06
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finguard_distilbert_37500_pipeline` is a English model originally trained by AnirudhLanka2002.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finguard_distilbert_37500_pipeline_en_5.5.0_3.0_1725653756300.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finguard_distilbert_37500_pipeline_en_5.5.0_3.0_1725653756300.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finguard_distilbert_37500_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finguard_distilbert_37500_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finguard_distilbert_37500_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|244.6 MB|

## References

https://huggingface.co/AnirudhLanka2002/finguard_distilBERT_37500

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification