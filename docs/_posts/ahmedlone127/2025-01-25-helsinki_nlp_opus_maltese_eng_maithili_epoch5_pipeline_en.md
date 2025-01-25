---
layout: model
title: English helsinki_nlp_opus_maltese_eng_maithili_epoch5_pipeline pipeline MarianTransformer from kubershahi
author: John Snow Labs
name: helsinki_nlp_opus_maltese_eng_maithili_epoch5_pipeline
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`helsinki_nlp_opus_maltese_eng_maithili_epoch5_pipeline` is a English model originally trained by kubershahi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/helsinki_nlp_opus_maltese_eng_maithili_epoch5_pipeline_en_5.5.1_3.0_1737781650279.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/helsinki_nlp_opus_maltese_eng_maithili_epoch5_pipeline_en_5.5.1_3.0_1737781650279.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("helsinki_nlp_opus_maltese_eng_maithili_epoch5_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("helsinki_nlp_opus_maltese_eng_maithili_epoch5_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|helsinki_nlp_opus_maltese_eng_maithili_epoch5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.4 MB|

## References

https://huggingface.co/kubershahi/Helsinki-NLP-opus-mt-eng-mai-epoch5

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer