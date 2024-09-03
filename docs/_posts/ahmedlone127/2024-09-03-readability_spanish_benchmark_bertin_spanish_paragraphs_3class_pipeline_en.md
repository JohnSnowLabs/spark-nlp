---
layout: model
title: English readability_spanish_benchmark_bertin_spanish_paragraphs_3class_pipeline pipeline RoBertaForSequenceClassification from lmvasque
author: John Snow Labs
name: readability_spanish_benchmark_bertin_spanish_paragraphs_3class_pipeline
date: 2024-09-03
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`readability_spanish_benchmark_bertin_spanish_paragraphs_3class_pipeline` is a English model originally trained by lmvasque.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/readability_spanish_benchmark_bertin_spanish_paragraphs_3class_pipeline_en_5.5.0_3.0_1725369456932.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/readability_spanish_benchmark_bertin_spanish_paragraphs_3class_pipeline_en_5.5.0_3.0_1725369456932.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("readability_spanish_benchmark_bertin_spanish_paragraphs_3class_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("readability_spanish_benchmark_bertin_spanish_paragraphs_3class_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|readability_spanish_benchmark_bertin_spanish_paragraphs_3class_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|464.5 MB|

## References

https://huggingface.co/lmvasque/readability-es-benchmark-bertin-es-paragraphs-3class

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification