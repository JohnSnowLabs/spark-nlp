---
layout: model
title: None kemmer_finetuned_russian_english_pipeline pipeline MarianTransformer from KemmerEdition
author: John Snow Labs
name: kemmer_finetuned_russian_english_pipeline
date: 2024-09-16
tags: [nan, open_source, pipeline, onnx]
task: Translation
language: nan
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kemmer_finetuned_russian_english_pipeline` is a None model originally trained by KemmerEdition.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kemmer_finetuned_russian_english_pipeline_nan_5.5.0_3.0_1726457047081.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kemmer_finetuned_russian_english_pipeline_nan_5.5.0_3.0_1726457047081.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kemmer_finetuned_russian_english_pipeline", lang = "nan")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kemmer_finetuned_russian_english_pipeline", lang = "nan")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kemmer_finetuned_russian_english_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|nan|
|Size:|526.9 MB|

## References

https://huggingface.co/KemmerEdition/Kemmer_Finetuned_Ru_En

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer