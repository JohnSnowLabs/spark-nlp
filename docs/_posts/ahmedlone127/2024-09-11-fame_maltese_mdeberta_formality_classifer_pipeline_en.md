---
layout: model
title: English fame_maltese_mdeberta_formality_classifer_pipeline pipeline DeBertaForSequenceClassification from laniqo
author: John Snow Labs
name: fame_maltese_mdeberta_formality_classifer_pipeline
date: 2024-09-11
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fame_maltese_mdeberta_formality_classifer_pipeline` is a English model originally trained by laniqo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fame_maltese_mdeberta_formality_classifer_pipeline_en_5.5.0_3.0_1726098403306.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fame_maltese_mdeberta_formality_classifer_pipeline_en_5.5.0_3.0_1726098403306.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fame_maltese_mdeberta_formality_classifer_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fame_maltese_mdeberta_formality_classifer_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fame_maltese_mdeberta_formality_classifer_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|817.4 MB|

## References

https://huggingface.co/laniqo/fame_mt_mdeberta_formality_classifer

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification