---
layout: model
title: English kanglish_offensive_language_identification_pipeline pipeline RoBertaForSequenceClassification from seanbenhur
author: John Snow Labs
name: kanglish_offensive_language_identification_pipeline
date: 2024-09-23
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kanglish_offensive_language_identification_pipeline` is a English model originally trained by seanbenhur.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kanglish_offensive_language_identification_pipeline_en_5.5.0_3.0_1727134938987.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kanglish_offensive_language_identification_pipeline_en_5.5.0_3.0_1727134938987.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kanglish_offensive_language_identification_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kanglish_offensive_language_identification_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kanglish_offensive_language_identification_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|451.8 MB|

## References

https://huggingface.co/seanbenhur/kanglish-offensive-language-identification

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification