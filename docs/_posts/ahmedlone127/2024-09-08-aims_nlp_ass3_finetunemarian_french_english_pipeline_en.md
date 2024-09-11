---
layout: model
title: English aims_nlp_ass3_finetunemarian_french_english_pipeline pipeline MarianTransformer from krame-aims
author: John Snow Labs
name: aims_nlp_ass3_finetunemarian_french_english_pipeline
date: 2024-09-08
tags: [en, open_source, pipeline, onnx]
task: Translation
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`aims_nlp_ass3_finetunemarian_french_english_pipeline` is a English model originally trained by krame-aims.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/aims_nlp_ass3_finetunemarian_french_english_pipeline_en_5.5.0_3.0_1725832072281.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/aims_nlp_ass3_finetunemarian_french_english_pipeline_en_5.5.0_3.0_1725832072281.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("aims_nlp_ass3_finetunemarian_french_english_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("aims_nlp_ass3_finetunemarian_french_english_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|aims_nlp_ass3_finetunemarian_french_english_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|508.1 MB|

## References

https://huggingface.co/krame-aims/AIMS-NLP-ASS3-FinetuneMarian-fr-en

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer