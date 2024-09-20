---
layout: model
title: English sbert_yelp2class_pipeline pipeline MPNetForSequenceClassification from Siki-77
author: John Snow Labs
name: sbert_yelp2class_pipeline
date: 2024-08-31
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained MPNetForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sbert_yelp2class_pipeline` is a English model originally trained by Siki-77.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sbert_yelp2class_pipeline_en_5.4.2_3.0_1725145617370.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sbert_yelp2class_pipeline_en_5.4.2_3.0_1725145617370.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sbert_yelp2class_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sbert_yelp2class_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbert_yelp2class_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.4 MB|

## References

https://huggingface.co/Siki-77/sbert_yelp2class

## Included Models

- DocumentAssembler
- TokenizerModel
- MPNetForSequenceClassification