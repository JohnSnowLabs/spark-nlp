---
layout: model
title: Persian nlp_ariabert_digimag_pipeline pipeline RoBertaForSequenceClassification from Arshia-HZ
author: John Snow Labs
name: nlp_ariabert_digimag_pipeline
date: 2025-02-03
tags: [fa, open_source, pipeline, onnx]
task: Text Classification
language: fa
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nlp_ariabert_digimag_pipeline` is a Persian model originally trained by Arshia-HZ.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nlp_ariabert_digimag_pipeline_fa_5.5.1_3.0_1738595207281.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nlp_ariabert_digimag_pipeline_fa_5.5.1_3.0_1738595207281.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nlp_ariabert_digimag_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nlp_ariabert_digimag_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nlp_ariabert_digimag_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|494.7 MB|

## References

https://huggingface.co/Arshia-HZ/NLP-AriaBert-Digimag

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification