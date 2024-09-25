---
layout: model
title: English quantum_neuraladaptivelearningsystem_pipeline pipeline BertForSequenceClassification from ayjays132
author: John Snow Labs
name: quantum_neuraladaptivelearningsystem_pipeline
date: 2024-09-25
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`quantum_neuraladaptivelearningsystem_pipeline` is a English model originally trained by ayjays132.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/quantum_neuraladaptivelearningsystem_pipeline_en_5.5.0_3.0_1727294042847.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/quantum_neuraladaptivelearningsystem_pipeline_en_5.5.0_3.0_1727294042847.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("quantum_neuraladaptivelearningsystem_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("quantum_neuraladaptivelearningsystem_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|quantum_neuraladaptivelearningsystem_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|410.1 MB|

## References

https://huggingface.co/ayjays132/Quantum-NeuralAdaptiveLearningSystem

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification