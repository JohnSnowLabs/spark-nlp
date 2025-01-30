---
layout: model
title: English turkish_zeroshot_distilbert_pipeline pipeline DistilBertForZeroShotClassification from kaixkhazaki
author: John Snow Labs
name: turkish_zeroshot_distilbert_pipeline
date: 2025-01-24
tags: [en, open_source, pipeline, onnx]
task: Zero-Shot Classification
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

Pretrained DistilBertForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`turkish_zeroshot_distilbert_pipeline` is a English model originally trained by kaixkhazaki.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/turkish_zeroshot_distilbert_pipeline_en_5.5.1_3.0_1737699993590.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/turkish_zeroshot_distilbert_pipeline_en_5.5.1_3.0_1737699993590.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("turkish_zeroshot_distilbert_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("turkish_zeroshot_distilbert_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|turkish_zeroshot_distilbert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|254.1 MB|

## References

https://huggingface.co/kaixkhazaki/turkish-zeroshot-distilbert

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForZeroShotClassification