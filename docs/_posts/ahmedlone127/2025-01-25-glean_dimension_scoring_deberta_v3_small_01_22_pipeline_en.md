---
layout: model
title: English glean_dimension_scoring_deberta_v3_small_01_22_pipeline pipeline DeBertaForSequenceClassification from withpi
author: John Snow Labs
name: glean_dimension_scoring_deberta_v3_small_01_22_pipeline
date: 2025-01-25
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`glean_dimension_scoring_deberta_v3_small_01_22_pipeline` is a English model originally trained by withpi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/glean_dimension_scoring_deberta_v3_small_01_22_pipeline_en_5.5.1_3.0_1737819993267.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/glean_dimension_scoring_deberta_v3_small_01_22_pipeline_en_5.5.1_3.0_1737819993267.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("glean_dimension_scoring_deberta_v3_small_01_22_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("glean_dimension_scoring_deberta_v3_small_01_22_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|glean_dimension_scoring_deberta_v3_small_01_22_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|418.0 MB|

## References

https://huggingface.co/withpi/glean_dimension_scoring_deberta-v3-small_01-22

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification