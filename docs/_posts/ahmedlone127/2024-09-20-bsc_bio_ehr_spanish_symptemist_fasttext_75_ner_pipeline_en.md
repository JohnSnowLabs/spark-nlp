---
layout: model
title: English bsc_bio_ehr_spanish_symptemist_fasttext_75_ner_pipeline pipeline RoBertaForTokenClassification from Rodrigo1771
author: John Snow Labs
name: bsc_bio_ehr_spanish_symptemist_fasttext_75_ner_pipeline
date: 2024-09-20
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bsc_bio_ehr_spanish_symptemist_fasttext_75_ner_pipeline` is a English model originally trained by Rodrigo1771.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bsc_bio_ehr_spanish_symptemist_fasttext_75_ner_pipeline_en_5.5.0_3.0_1726847503107.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bsc_bio_ehr_spanish_symptemist_fasttext_75_ner_pipeline_en_5.5.0_3.0_1726847503107.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bsc_bio_ehr_spanish_symptemist_fasttext_75_ner_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bsc_bio_ehr_spanish_symptemist_fasttext_75_ner_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bsc_bio_ehr_spanish_symptemist_fasttext_75_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|435.7 MB|

## References

https://huggingface.co/Rodrigo1771/bsc-bio-ehr-es-symptemist-fasttext-75-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification