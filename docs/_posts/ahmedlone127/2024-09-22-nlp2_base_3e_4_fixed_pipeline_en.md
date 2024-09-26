---
layout: model
title: English nlp2_base_3e_4_fixed_pipeline pipeline DistilBertForSequenceClassification from VRT-2428211
author: John Snow Labs
name: nlp2_base_3e_4_fixed_pipeline
date: 2024-09-22
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nlp2_base_3e_4_fixed_pipeline` is a English model originally trained by VRT-2428211.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nlp2_base_3e_4_fixed_pipeline_en_5.5.0_3.0_1727033727339.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nlp2_base_3e_4_fixed_pipeline_en_5.5.0_3.0_1727033727339.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nlp2_base_3e_4_fixed_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nlp2_base_3e_4_fixed_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nlp2_base_3e_4_fixed_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/VRT-2428211/NLP2_Base_3e-4_Fixed

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification