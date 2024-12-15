---
layout: model
title: English toxicity_token_classifier_pipeline pipeline MPNetForTokenClassification from Sinanmz
author: John Snow Labs
name: toxicity_token_classifier_pipeline
date: 2024-12-15
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

Pretrained MPNetForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`toxicity_token_classifier_pipeline` is a English model originally trained by Sinanmz.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/toxicity_token_classifier_pipeline_en_5.5.1_3.0_1734290809123.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/toxicity_token_classifier_pipeline_en_5.5.1_3.0_1734290809123.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("toxicity_token_classifier_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("toxicity_token_classifier_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|toxicity_token_classifier_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|290.2 MB|

## References

https://huggingface.co/Sinanmz/toxicity_token_classifier

## Included Models

- DocumentAssembler
- TokenizerModel
- MPNetForTokenClassification