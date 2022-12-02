---
layout: model
title: Clinical Deidentification
author: John Snow Labs
name: clinical_deidentification
date: 2022-01-06
tags: [deidentification, licensed, pipeline, de]
task: Pipeline Healthcare
language: de
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pipeline can be used to deidentify PHI information from **German** medical texts. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask and obfuscate `PATIENT`, `HOSPITAL`, `DATE`, `ORGANIZATION`, `CITY`, `STREET`, `USERNAME`, `PROFESSION`, `PHONE`, `COUNTRY`, `DOCTOR`, `AGE`, `CONTACT`, `ID`, `PHONE`, `ZIP`, `ACCOUNT`, `SSN`, `DLN`, `PLATE` entities.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_de_3.3.4_2.4_1641504439647.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
from sparknlp.pretrained import PretrainedPipeline

deid_pipeline = PretrainedPipeline("clinical_deidentification", "de", "clinical/models")

sample = """Michael Berger wird am Morgen des 12 Dezember 2018 ins St. Elisabeth-Krankenhaus
in Bad Kissingen eingeliefert. Herr Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen."""

result = deid_pipe.annotate(sample)

print("\n".join(result['masked']))
print("\n".join(result['obfuscated']))
```
```scala
...
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val deid_pipeline = PretrainedPipeline("clinical_deidentification","de","clinical/models")

val sample = """Michael Berger wird am Morgen des 12 Dezember 2018 ins St. Elisabeth-Krankenhaus
in Bad Kissingen eingeliefert. Herr Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen."""

val result = deid_pipe.annotate(sample)
```
</div>

## Results

```bash
<PATIENT> wird am Morgen des <DATE> ins <HOSPITAL> in <CITY> eingeliefert.
<PATIENT> ist <AGE> Jahre alt und hat zu viel Wasser in den Beinen.

Mathias Farber wird am Morgen des 05-01-1978 ins Rechts der Isar Hospital in Berlin eingeliefert.
Mathias Farber ist 56 Jahre alt und hat zu viel Wasser in den Beinen.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clinical_deidentification|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Language:|de|
|Size:|1.3 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
- ContextualParserModel
- ChunkMergeModel
- DeIdentificationModel
