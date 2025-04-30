---
layout: model
title: Portuguese tempclin_biobertpt_bio_pipeline pipeline BertForTokenClassification from pucpr-br
author: John Snow Labs
name: tempclin_biobertpt_bio_pipeline
date: 2025-04-04
tags: [pt, open_source, pipeline, onnx]
task: Named Entity Recognition
language: pt
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tempclin_biobertpt_bio_pipeline` is a Portuguese model originally trained by pucpr-br.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tempclin_biobertpt_bio_pipeline_pt_5.5.1_3.0_1743744814954.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tempclin_biobertpt_bio_pipeline_pt_5.5.1_3.0_1743744814954.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tempclin_biobertpt_bio_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tempclin_biobertpt_bio_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tempclin_biobertpt_bio_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|665.1 MB|

## References

https://huggingface.co/pucpr-br/tempclin-biobertpt-bio

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification