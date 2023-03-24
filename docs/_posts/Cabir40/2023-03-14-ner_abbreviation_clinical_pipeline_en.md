---
layout: model
title: Pipeline to Extraction of Clinical Abbreviations and Acronyms
author: John Snow Labs
name: ner_abbreviation_clinical_pipeline
date: 2023-03-14
tags: [ner, abbreviation, acronym, en, clinical, licensed]
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

This pretrained pipeline is built on the top of [ner_abbreviation_clinical](https://nlp.johnsnowlabs.com/2021/12/30/ner_abbreviation_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_abbreviation_clinical_pipeline_en_4.3.0_3.2_1678777406281.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_abbreviation_clinical_pipeline_en_4.3.0_3.2_1678777406281.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_abbreviation_clinical_pipeline", "en", "clinical/models")

text = '''Gravid with estimated fetal weight of 6-6/12 pounds. LOWER EXTREMITIES: No edema. LABORATORY DATA: Laboratory tests include a CBC which is normal. Blood Type: AB positive. Rubella: Immune. VDRL: Nonreactive. Hepatitis C surface antigen: Negative. HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_abbreviation_clinical_pipeline", "en", "clinical/models")

val text = "Gravid with estimated fetal weight of 6-6/12 pounds. LOWER EXTREMITIES: No edema. LABORATORY DATA: Laboratory tests include a CBC which is normal. Blood Type: AB positive. Rubella: Immune. VDRL: Nonreactive. Hepatitis C surface antigen: Negative. HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks   |   begin |   end | ner_label   |   confidence |
|---:|:-------------|--------:|------:|:------------|-------------:|
|  0 | CBC          |     126 |   128 | ABBR        |            1 |
|  1 | AB           |     159 |   160 | ABBR        |            1 |
|  2 | VDRL         |     189 |   192 | ABBR        |            1 |
|  3 | HIV          |     247 |   249 | ABBR        |            1 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_abbreviation_clinical_pipeline|
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