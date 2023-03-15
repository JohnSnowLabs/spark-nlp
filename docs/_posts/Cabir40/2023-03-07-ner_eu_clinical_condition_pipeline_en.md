---
layout: model
title: Pipeline to Detect Clinical Conditions (ner_eu_clinical_condition)
author: John Snow Labs
name: ner_eu_clinical_condition_pipeline
date: 2023-03-07
tags: [en, clinical, licensed, ner]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_eu_clinical_condition](https://nlp.johnsnowlabs.com/2023/02/06/ner_eu_clinical_condition_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_condition_pipeline_en_4.3.0_3.2_1678213988790.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_condition_pipeline_en_4.3.0_3.2_1678213988790.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_eu_clinical_condition_pipeline", "en", "clinical/models")

text = "
Hyperparathyroidism was considered upon the fourth occasion. The history of weakness and generalized joint pains were present. He also had history of epigastric pain diagnosed informally as gastritis. He had previously had open reduction and internal fixation for the initial two fractures under general anesthesia. He sustained mandibular fracture.
"

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_eu_clinical_condition_pipeline", "en", "clinical/models")

val text = "
Hyperparathyroidism was considered upon the fourth occasion. The history of weakness and generalized joint pains were present. He also had history of epigastric pain diagnosed informally as gastritis. He had previously had open reduction and internal fixation for the initial two fractures under general anesthesia. He sustained mandibular fracture.
"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | chunks                  |   begin |   end | entities           |   confidence |
|---:|:------------------------|--------:|------:|:-------------------|-------------:|
|  0 | Hyperparathyroidism     |       1 |    19 | clinical_condition |     0.9375   |
|  1 | weakness                |      77 |    84 | clinical_condition |     0.9779   |
|  2 | generalized joint pains |      90 |   112 | clinical_condition |     0.717333 |
|  3 | epigastric pain         |     151 |   165 | clinical_condition |     0.64985  |
|  4 | gastritis               |     191 |   199 | clinical_condition |     0.9543   |
|  5 | fractures               |     281 |   289 | clinical_condition |     0.9726   |
|  6 | anesthesia              |     305 |   314 | clinical_condition |     0.991    |
|  7 | mandibular fracture     |     330 |   348 | clinical_condition |     0.54925  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_eu_clinical_condition_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel