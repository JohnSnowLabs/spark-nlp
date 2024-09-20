---
layout: model
title: Multilingual distilbert_base_multilingual_cased_jaquad_pipeline pipeline DistilBertForQuestionAnswering from cuongtk2002
author: John Snow Labs
name: distilbert_base_multilingual_cased_jaquad_pipeline
date: 2024-09-15
tags: [xx, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained DistilBertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_base_multilingual_cased_jaquad_pipeline` is a Multilingual model originally trained by cuongtk2002.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_multilingual_cased_jaquad_pipeline_xx_5.5.0_3.0_1726382538372.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_base_multilingual_cased_jaquad_pipeline_xx_5.5.0_3.0_1726382538372.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilbert_base_multilingual_cased_jaquad_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilbert_base_multilingual_cased_jaquad_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_multilingual_cased_jaquad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|505.4 MB|

## References

https://huggingface.co/cuongtk2002/distilbert-base-multilingual-cased-JaQuAD

## Included Models

- MultiDocumentAssembler
- DistilBertForQuestionAnswering