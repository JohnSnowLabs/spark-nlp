---
layout: model
title: Arabic arabert_ner_pipeline pipeline BertForTokenClassification from abdusah
author: John Snow Labs
name: arabert_ner_pipeline
date: 2024-09-05
tags: [ar, open_source, pipeline, onnx]
task: Named Entity Recognition
language: ar
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`arabert_ner_pipeline` is a Arabic model originally trained by abdusah.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/arabert_ner_pipeline_ar_5.5.0_3.0_1725564098471.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/arabert_ner_pipeline_ar_5.5.0_3.0_1725564098471.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("arabert_ner_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("arabert_ner_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|arabert_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|504.9 MB|

## References

https://huggingface.co/abdusah/arabert-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification