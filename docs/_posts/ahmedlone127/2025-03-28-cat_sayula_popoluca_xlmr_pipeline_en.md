---
layout: model
title: English cat_sayula_popoluca_xlmr_pipeline pipeline XlmRoBertaForTokenClassification from homersimpson
author: John Snow Labs
name: cat_sayula_popoluca_xlmr_pipeline
date: 2025-03-28
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cat_sayula_popoluca_xlmr_pipeline` is a English model originally trained by homersimpson.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cat_sayula_popoluca_xlmr_pipeline_en_5.5.1_3.0_1743123740554.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cat_sayula_popoluca_xlmr_pipeline_en_5.5.1_3.0_1743123740554.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cat_sayula_popoluca_xlmr_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cat_sayula_popoluca_xlmr_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cat_sayula_popoluca_xlmr_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|815.8 MB|

## References

https://huggingface.co/homersimpson/cat-pos-xlmr

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification