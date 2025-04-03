---
layout: model
title: English squeezebert_finetuned_squadv1_pipeline pipeline BertForQuestionAnswering from mrm8488
author: John Snow Labs
name: squeezebert_finetuned_squadv1_pipeline
date: 2025-04-03
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`squeezebert_finetuned_squadv1_pipeline` is a English model originally trained by mrm8488.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/squeezebert_finetuned_squadv1_pipeline_en_5.5.1_3.0_1743647316314.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/squeezebert_finetuned_squadv1_pipeline_en_5.5.1_3.0_1743647316314.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("squeezebert_finetuned_squadv1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("squeezebert_finetuned_squadv1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|squeezebert_finetuned_squadv1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|186.4 MB|

## References

https://huggingface.co/mrm8488/squeezebert-finetuned-squadv1

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering