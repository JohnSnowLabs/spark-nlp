---
layout: model
title: German roberta_fine_tune_german_ner_pipeline pipeline RoBertaForTokenClassification from MAbokahf
author: John Snow Labs
name: roberta_fine_tune_german_ner_pipeline
date: 2024-09-17
tags: [de, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_fine_tune_german_ner_pipeline` is a German model originally trained by MAbokahf.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_fine_tune_german_ner_pipeline_de_5.5.0_3.0_1726537699453.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_fine_tune_german_ner_pipeline_de_5.5.0_3.0_1726537699453.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_fine_tune_german_ner_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_fine_tune_german_ner_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_fine_tune_german_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|433.3 MB|

## References

https://huggingface.co/MAbokahf/roberta-fine-tune-de-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification