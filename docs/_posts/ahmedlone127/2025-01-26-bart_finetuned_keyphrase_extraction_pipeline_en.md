---
layout: model
title: English bart_finetuned_keyphrase_extraction_pipeline pipeline BartTransformer from aglazkova
author: John Snow Labs
name: bart_finetuned_keyphrase_extraction_pipeline
date: 2025-01-26
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
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

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bart_finetuned_keyphrase_extraction_pipeline` is a English model originally trained by aglazkova.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bart_finetuned_keyphrase_extraction_pipeline_en_5.5.1_3.0_1737901843247.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bart_finetuned_keyphrase_extraction_pipeline_en_5.5.1_3.0_1737901843247.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bart_finetuned_keyphrase_extraction_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bart_finetuned_keyphrase_extraction_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bart_finetuned_keyphrase_extraction_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|809.6 MB|

## References

https://huggingface.co/aglazkova/bart_finetuned_keyphrase_extraction

## Included Models

- DocumentAssembler
- BartTransformer