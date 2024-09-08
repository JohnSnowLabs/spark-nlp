---
layout: model
title: Portuguese bertimbau_large_ner_total_pipeline pipeline BertForTokenClassification from marquesafonso
author: John Snow Labs
name: bertimbau_large_ner_total_pipeline
date: 2024-09-04
tags: [pt, open_source, pipeline, onnx]
task: Named Entity Recognition
language: pt
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bertimbau_large_ner_total_pipeline` is a Portuguese model originally trained by marquesafonso.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bertimbau_large_ner_total_pipeline_pt_5.5.0_3.0_1725477385153.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bertimbau_large_ner_total_pipeline_pt_5.5.0_3.0_1725477385153.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bertimbau_large_ner_total_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bertimbau_large_ner_total_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bertimbau_large_ner_total_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|406.0 MB|

## References

https://huggingface.co/marquesafonso/bertimbau-large-ner-total

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification