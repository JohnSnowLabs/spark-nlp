---
layout: model
title: English analisis_sentimientos_xlm_roberta_tass_pipeline pipeline XlmRoBertaForSequenceClassification from raulgdp
author: John Snow Labs
name: analisis_sentimientos_xlm_roberta_tass_pipeline
date: 2025-01-31
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`analisis_sentimientos_xlm_roberta_tass_pipeline` is a English model originally trained by raulgdp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/analisis_sentimientos_xlm_roberta_tass_pipeline_en_5.5.1_3.0_1738340281000.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/analisis_sentimientos_xlm_roberta_tass_pipeline_en_5.5.1_3.0_1738340281000.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("analisis_sentimientos_xlm_roberta_tass_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("analisis_sentimientos_xlm_roberta_tass_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|analisis_sentimientos_xlm_roberta_tass_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.0 GB|

## References

https://huggingface.co/raulgdp/Analisis-sentimientos-XLM-Roberta-TASS

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification