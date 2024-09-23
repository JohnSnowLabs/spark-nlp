---
layout: model
title: Modern Greek (1453-) humor_recognition_greek_distilbert_pipeline pipeline DistilBertForSequenceClassification from Kalloniatis
author: John Snow Labs
name: humor_recognition_greek_distilbert_pipeline
date: 2024-09-23
tags: [el, open_source, pipeline, onnx]
task: Text Classification
language: el
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`humor_recognition_greek_distilbert_pipeline` is a Modern Greek (1453-) model originally trained by Kalloniatis.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/humor_recognition_greek_distilbert_pipeline_el_5.5.0_3.0_1727074201489.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/humor_recognition_greek_distilbert_pipeline_el_5.5.0_3.0_1727074201489.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("humor_recognition_greek_distilbert_pipeline", lang = "el")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("humor_recognition_greek_distilbert_pipeline", lang = "el")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|humor_recognition_greek_distilbert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|el|
|Size:|507.6 MB|

## References

https://huggingface.co/Kalloniatis/Humor-Recognition-Greek-DistilBERT

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification