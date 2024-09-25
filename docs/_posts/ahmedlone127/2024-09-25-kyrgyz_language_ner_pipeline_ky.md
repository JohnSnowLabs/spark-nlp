---
layout: model
title: Kirghiz, Kyrgyz kyrgyz_language_ner_pipeline pipeline BertForTokenClassification from murat
author: John Snow Labs
name: kyrgyz_language_ner_pipeline
date: 2024-09-25
tags: [ky, open_source, pipeline, onnx]
task: Named Entity Recognition
language: ky
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kyrgyz_language_ner_pipeline` is a Kirghiz, Kyrgyz model originally trained by murat.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kyrgyz_language_ner_pipeline_ky_5.5.0_3.0_1727249950446.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kyrgyz_language_ner_pipeline_ky_5.5.0_3.0_1727249950446.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kyrgyz_language_ner_pipeline", lang = "ky")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kyrgyz_language_ner_pipeline", lang = "ky")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kyrgyz_language_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ky|
|Size:|665.1 MB|

## References

https://huggingface.co/murat/kyrgyz_language_NER

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification