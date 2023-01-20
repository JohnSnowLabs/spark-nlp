---
layout: model
title: Pipeline to Detect Temporal Relations for Clinical Events (Enriched)
author: John Snow Labs
name: re_temporal_events_enriched_clinical_pipeline
date: 2022-03-31
tags: [licensed, clinical, relation_extraction, event, enriched, en]
task: Relation Extraction
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [re_temporal_events_enriched_clinical](https://nlp.johnsnowlabs.com/2020/09/28/re_temporal_events_enriched_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_temporal_events_enriched_clinical_pipeline_en_3.4.1_3.0_1648734605627.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/re_temporal_events_enriched_clinical_pipeline_en_3.4.1_3.0_1648734605627.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("re_temporal_events_enriched_clinical_pipeline", "en", "clinical/models")


pipeline.annotate("The patient is a 56-year-old right-handed female with longstanding intermittent right low back pain, who was involved in a motor vehicle accident in September of 2005. At that time, she did not notice any specific injury, but five days later, she started getting abnormal right low back pain.")
```
```scala
val pipeline = new PretrainedPipeline("re_temporal_events_enriched_clinical_pipeline", "en", "clinical/models")


pipeline.annotate("The patient is a 56-year-old right-handed female with longstanding intermittent right low back pain, who was involved in a motor vehicle accident in September of 2005. At that time, she did not notice any specific injury, but five days later, she started getting abnormal right low back pain.")
```
</div>

## Results

```bash
+----+------------+-----------+-----------------+---------------+-----------------------------------------------+------------+-----------------+---------------+--------------------------+--------------+
|    | relation   | entity1   |   entity1_begin |   entity1_end | chunk1                                        | entity2    |   entity2_begin |   entity2_end | chunk2                   |   confidence |
+====+============+===========+=================+===============+===============================================+============+=================+===============+==========================+==============+
|  0 | OVERLAP    | PROBLEM   |              54 |            98 | longstanding intermittent right low back pain | OCCURRENCE |             121 |           144 | a motor vehicle accident |     0.532308 |
+----+------------+-----------+-----------------+---------------+-----------------------------------------------+------------+-----------------+---------------+--------------------------+--------------+
|  1 | AFTER      | DATE      |             171 |           179 | that time                                     | PROBLEM    |             201 |           219 | any specific injury      |     0.577288 |
+----+------------+-----------+-----------------+---------------+-----------------------------------------------+------------+-----------------+---------------+--------------------------+--------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_temporal_events_enriched_clinical_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- PerceptronModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
- DependencyParserModel
- RelationExtractionModel