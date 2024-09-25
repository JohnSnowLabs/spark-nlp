---
layout: model
title: English roberta_spanish_clinical_trials_medic_attr_ner_pipeline pipeline RoBertaForTokenClassification from medspaner
author: John Snow Labs
name: roberta_spanish_clinical_trials_medic_attr_ner_pipeline
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_spanish_clinical_trials_medic_attr_ner_pipeline` is a English model originally trained by medspaner.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_spanish_clinical_trials_medic_attr_ner_pipeline_en_5.5.0_3.0_1726653453128.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_spanish_clinical_trials_medic_attr_ner_pipeline_en_5.5.0_3.0_1726653453128.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_spanish_clinical_trials_medic_attr_ner_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_spanish_clinical_trials_medic_attr_ner_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_spanish_clinical_trials_medic_attr_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|442.3 MB|

## References

https://huggingface.co/medspaner/roberta-es-clinical-trials-medic-attr-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification