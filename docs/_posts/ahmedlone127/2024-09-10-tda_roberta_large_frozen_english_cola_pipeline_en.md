---
layout: model
title: English tda_roberta_large_frozen_english_cola_pipeline pipeline RoBertaForSequenceClassification from iproskurina
author: John Snow Labs
name: tda_roberta_large_frozen_english_cola_pipeline
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tda_roberta_large_frozen_english_cola_pipeline` is a English model originally trained by iproskurina.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tda_roberta_large_frozen_english_cola_pipeline_en_5.5.0_3.0_1725972044945.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tda_roberta_large_frozen_english_cola_pipeline_en_5.5.0_3.0_1725972044945.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tda_roberta_large_frozen_english_cola_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tda_roberta_large_frozen_english_cola_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tda_roberta_large_frozen_english_cola_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|846.5 MB|

## References

https://huggingface.co/iproskurina/tda-roberta-large-frozen-en-cola

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification