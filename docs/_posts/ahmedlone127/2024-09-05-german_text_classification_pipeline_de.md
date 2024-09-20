---
layout: model
title: German german_text_classification_pipeline pipeline XlmRoBertaForSequenceClassification from RashidNLP
author: John Snow Labs
name: german_text_classification_pipeline
date: 2024-09-05
tags: [de, open_source, pipeline, onnx]
task: Text Classification
language: de
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`german_text_classification_pipeline` is a German model originally trained by RashidNLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/german_text_classification_pipeline_de_5.5.0_3.0_1725529174863.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/german_text_classification_pipeline_de_5.5.0_3.0_1725529174863.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("german_text_classification_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("german_text_classification_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|german_text_classification_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|779.2 MB|

## References

https://huggingface.co/RashidNLP/German-Text-Classification

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification