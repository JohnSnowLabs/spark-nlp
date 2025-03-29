---
layout: model
title: English finetuned_helsinki_nlp_english_marathi_marh_pipeline pipeline MarianTransformer from anujsahani01
author: John Snow Labs
name: finetuned_helsinki_nlp_english_marathi_marh_pipeline
date: 2025-03-29
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_helsinki_nlp_english_marathi_marh_pipeline` is a English model originally trained by anujsahani01.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_helsinki_nlp_english_marathi_marh_pipeline_en_5.5.1_3.0_1743246979828.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_helsinki_nlp_english_marathi_marh_pipeline_en_5.5.1_3.0_1743246979828.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("finetuned_helsinki_nlp_english_marathi_marh_pipeline", lang = "en")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("finetuned_helsinki_nlp_english_marathi_marh_pipeline", lang = "en")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetuned_helsinki_nlp_english_marathi_marh_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|405.3 MB|

## References

References

https://huggingface.co/anujsahani01/finetuned_Helsinki-NLP-en-mr

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer