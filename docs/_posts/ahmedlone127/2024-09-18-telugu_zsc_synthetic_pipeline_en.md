---
layout: model
title: English telugu_zsc_synthetic_pipeline pipeline RoBertaForSequenceClassification from adriansanz
author: John Snow Labs
name: telugu_zsc_synthetic_pipeline
date: 2024-09-18
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`telugu_zsc_synthetic_pipeline` is a English model originally trained by adriansanz.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/telugu_zsc_synthetic_pipeline_en_5.5.0_3.0_1726650234767.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/telugu_zsc_synthetic_pipeline_en_5.5.0_3.0_1726650234767.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("telugu_zsc_synthetic_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("telugu_zsc_synthetic_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|telugu_zsc_synthetic_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|449.5 MB|

## References

https://huggingface.co/adriansanz/te-zsc-synthetic

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification