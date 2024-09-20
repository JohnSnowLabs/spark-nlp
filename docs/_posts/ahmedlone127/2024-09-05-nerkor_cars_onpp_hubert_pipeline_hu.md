---
layout: model
title: Hungarian nerkor_cars_onpp_hubert_pipeline pipeline BertForTokenClassification from novakat
author: John Snow Labs
name: nerkor_cars_onpp_hubert_pipeline
date: 2024-09-05
tags: [hu, open_source, pipeline, onnx]
task: Named Entity Recognition
language: hu
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nerkor_cars_onpp_hubert_pipeline` is a Hungarian model originally trained by novakat.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nerkor_cars_onpp_hubert_pipeline_hu_5.5.0_3.0_1725511550361.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nerkor_cars_onpp_hubert_pipeline_hu_5.5.0_3.0_1725511550361.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nerkor_cars_onpp_hubert_pipeline", lang = "hu")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nerkor_cars_onpp_hubert_pipeline", lang = "hu")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nerkor_cars_onpp_hubert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|hu|
|Size:|412.7 MB|

## References

https://huggingface.co/novakat/nerkor-cars-onpp-hubert

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification