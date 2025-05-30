---
layout: model
title: Multilingual indobert_lite_base_p1_indonli_multilingual_nli_distil_mdeberta_pipeline pipeline BertForSequenceClassification from LazarusNLP
author: John Snow Labs
name: indobert_lite_base_p1_indonli_multilingual_nli_distil_mdeberta_pipeline
date: 2024-09-12
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`indobert_lite_base_p1_indonli_multilingual_nli_distil_mdeberta_pipeline` is a Multilingual model originally trained by LazarusNLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/indobert_lite_base_p1_indonli_multilingual_nli_distil_mdeberta_pipeline_xx_5.5.0_3.0_1726104383168.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/indobert_lite_base_p1_indonli_multilingual_nli_distil_mdeberta_pipeline_xx_5.5.0_3.0_1726104383168.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("indobert_lite_base_p1_indonli_multilingual_nli_distil_mdeberta_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("indobert_lite_base_p1_indonli_multilingual_nli_distil_mdeberta_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|indobert_lite_base_p1_indonli_multilingual_nli_distil_mdeberta_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|44.1 MB|

## References

https://huggingface.co/LazarusNLP/indobert-lite-base-p1-indonli-multilingual-nli-distil-mdeberta

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification