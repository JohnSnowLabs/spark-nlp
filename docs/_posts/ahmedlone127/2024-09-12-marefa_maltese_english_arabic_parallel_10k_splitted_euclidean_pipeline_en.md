---
layout: model
title: English marefa_maltese_english_arabic_parallel_10k_splitted_euclidean_pipeline pipeline MarianTransformer from HamdanXI
author: John Snow Labs
name: marefa_maltese_english_arabic_parallel_10k_splitted_euclidean_pipeline
date: 2024-09-12
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`marefa_maltese_english_arabic_parallel_10k_splitted_euclidean_pipeline` is a English model originally trained by HamdanXI.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/marefa_maltese_english_arabic_parallel_10k_splitted_euclidean_pipeline_en_5.5.0_3.0_1726126819339.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/marefa_maltese_english_arabic_parallel_10k_splitted_euclidean_pipeline_en_5.5.0_3.0_1726126819339.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("marefa_maltese_english_arabic_parallel_10k_splitted_euclidean_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("marefa_maltese_english_arabic_parallel_10k_splitted_euclidean_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|marefa_maltese_english_arabic_parallel_10k_splitted_euclidean_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|528.0 MB|

## References

https://huggingface.co/HamdanXI/marefa-mt-en-ar-parallel-10k-splitted-euclidean

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer