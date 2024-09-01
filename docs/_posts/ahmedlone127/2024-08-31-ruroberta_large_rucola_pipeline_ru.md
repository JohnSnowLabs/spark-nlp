---
layout: model
title: Russian ruroberta_large_rucola_pipeline pipeline RoBertaForSequenceClassification from RussianNLP
author: John Snow Labs
name: ruroberta_large_rucola_pipeline
date: 2024-08-31
tags: [ru, open_source, pipeline, onnx]
task: Text Classification
language: ru
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ruroberta_large_rucola_pipeline` is a Russian model originally trained by RussianNLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ruroberta_large_rucola_pipeline_ru_5.4.2_3.0_1725119155982.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ruroberta_large_rucola_pipeline_ru_5.4.2_3.0_1725119155982.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ruroberta_large_rucola_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ruroberta_large_rucola_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ruroberta_large_rucola_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|1.3 GB|

## References

https://huggingface.co/RussianNLP/ruRoBERTa-large-rucola

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification