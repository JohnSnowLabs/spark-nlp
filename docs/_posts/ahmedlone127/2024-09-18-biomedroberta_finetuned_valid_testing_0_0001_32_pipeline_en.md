---
layout: model
title: English biomedroberta_finetuned_valid_testing_0_0001_32_pipeline pipeline RoBertaForTokenClassification from pabRomero
author: John Snow Labs
name: biomedroberta_finetuned_valid_testing_0_0001_32_pipeline
date: 2024-09-18
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`biomedroberta_finetuned_valid_testing_0_0001_32_pipeline` is a English model originally trained by pabRomero.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biomedroberta_finetuned_valid_testing_0_0001_32_pipeline_en_5.5.0_3.0_1726652676147.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/biomedroberta_finetuned_valid_testing_0_0001_32_pipeline_en_5.5.0_3.0_1726652676147.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("biomedroberta_finetuned_valid_testing_0_0001_32_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("biomedroberta_finetuned_valid_testing_0_0001_32_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|biomedroberta_finetuned_valid_testing_0_0001_32_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|466.4 MB|

## References

https://huggingface.co/pabRomero/BioMedRoBERTa-finetuned-valid-testing-0.0001-32

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification