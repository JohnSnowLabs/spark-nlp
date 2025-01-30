---
layout: model
title: English finetuning_maltese_english_arabic_english_arabic_translation_pipeline pipeline MarianTransformer from ahmed792002
author: John Snow Labs
name: finetuning_maltese_english_arabic_english_arabic_translation_pipeline
date: 2025-01-25
tags: [en, open_source, pipeline, onnx]
task: Translation
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuning_maltese_english_arabic_english_arabic_translation_pipeline` is a English model originally trained by ahmed792002.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuning_maltese_english_arabic_english_arabic_translation_pipeline_en_5.5.1_3.0_1737829990140.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuning_maltese_english_arabic_english_arabic_translation_pipeline_en_5.5.1_3.0_1737829990140.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetuning_maltese_english_arabic_english_arabic_translation_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetuning_maltese_english_arabic_english_arabic_translation_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetuning_maltese_english_arabic_english_arabic_translation_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|410.0 MB|

## References

https://huggingface.co/ahmed792002/Finetuning_mt-en-ar_English_Arabic_Translation

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer