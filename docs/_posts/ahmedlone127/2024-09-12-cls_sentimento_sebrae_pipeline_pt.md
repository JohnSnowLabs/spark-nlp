---
layout: model
title: Portuguese cls_sentimento_sebrae_pipeline pipeline BertForSequenceClassification from ggrazzioli
author: John Snow Labs
name: cls_sentimento_sebrae_pipeline
date: 2024-09-12
tags: [pt, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cls_sentimento_sebrae_pipeline` is a Portuguese model originally trained by ggrazzioli.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cls_sentimento_sebrae_pipeline_pt_5.5.0_3.0_1726182474392.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cls_sentimento_sebrae_pipeline_pt_5.5.0_3.0_1726182474392.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cls_sentimento_sebrae_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cls_sentimento_sebrae_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cls_sentimento_sebrae_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|408.2 MB|

## References

https://huggingface.co/ggrazzioli/cls_sentimento_sebrae

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification