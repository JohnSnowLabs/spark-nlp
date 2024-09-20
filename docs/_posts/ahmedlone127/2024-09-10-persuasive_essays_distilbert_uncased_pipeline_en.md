---
layout: model
title: English persuasive_essays_distilbert_uncased_pipeline pipeline DistilBertForSequenceClassification from przvl
author: John Snow Labs
name: persuasive_essays_distilbert_uncased_pipeline
date: 2024-09-10
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`persuasive_essays_distilbert_uncased_pipeline` is a English model originally trained by przvl.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/persuasive_essays_distilbert_uncased_pipeline_en_5.5.0_3.0_1726009321632.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/persuasive_essays_distilbert_uncased_pipeline_en_5.5.0_3.0_1726009321632.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("persuasive_essays_distilbert_uncased_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("persuasive_essays_distilbert_uncased_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|persuasive_essays_distilbert_uncased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/przvl/persuasive_essays_distilbert_uncased

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification