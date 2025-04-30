---
layout: model
title: English opus_maltese_english_tok_pisin_finetune_2025_03_05_pipeline pipeline MarianTransformer from RickBrannan
author: John Snow Labs
name: opus_maltese_english_tok_pisin_finetune_2025_03_05_pipeline
date: 2025-03-28
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`opus_maltese_english_tok_pisin_finetune_2025_03_05_pipeline` is a English model originally trained by RickBrannan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/opus_maltese_english_tok_pisin_finetune_2025_03_05_pipeline_en_5.5.1_3.0_1743194328102.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/opus_maltese_english_tok_pisin_finetune_2025_03_05_pipeline_en_5.5.1_3.0_1743194328102.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("opus_maltese_english_tok_pisin_finetune_2025_03_05_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("opus_maltese_english_tok_pisin_finetune_2025_03_05_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|opus_maltese_english_tok_pisin_finetune_2025_03_05_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|357.1 MB|

## References

https://huggingface.co/RickBrannan/opus-mt-en-tpi-finetune-2025-03-05

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer