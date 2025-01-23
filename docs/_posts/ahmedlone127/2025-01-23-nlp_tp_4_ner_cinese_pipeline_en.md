---
layout: model
title: English nlp_tp_4_ner_cinese_pipeline pipeline DistilBertForTokenClassification from marcocarotta
author: John Snow Labs
name: nlp_tp_4_ner_cinese_pipeline
date: 2025-01-23
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nlp_tp_4_ner_cinese_pipeline` is a English model originally trained by marcocarotta.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nlp_tp_4_ner_cinese_pipeline_en_5.5.1_3.0_1737627349827.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nlp_tp_4_ner_cinese_pipeline_en_5.5.1_3.0_1737627349827.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nlp_tp_4_ner_cinese_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nlp_tp_4_ner_cinese_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nlp_tp_4_ner_cinese_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.3 MB|

## References

https://huggingface.co/marcocarotta/NLP-tp-4-ner-cinese

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification