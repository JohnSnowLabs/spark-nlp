---
layout: model
title: Clinical Deidentification
author: John Snow Labs
name: clinical_deidentification
date: 2022-01-08
tags: [deidentification, licensed, pipeline, de]
task: Pipeline Healthcare
language: de
edition: Healthcare NLP 3.4.0
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_de_3.4.0_2.4_1641636618956.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_de_3.4.0_2.4_1641636618956.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

deid_pipeline = PretrainedPipeline("clinical_deidentification", "de", "clinical/models")

sample = """Zusammenfassung : Michael Berger wird am Morgen des 12 Dezember 2018 ins St.Elisabeth Krankenhaus eingeliefert. 
Herr Michael Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen.

Persönliche Daten :
ID-Nummer: T0110053F
Platte A-BC124
Kontonummer: DE89370400440532013000
SSN : 13110587M565
Lizenznummer: B072RRE2I55
Adresse : St.Johann-Straße 13 19300
"""

result = deid_pipeline.annotate(sample)
print("\n".join(result['masked']))
print("\n".join(result['masked_with_chars']))
print("\n".join(result['masked_fixed_length_chars']))
print("\n".join(result['obfuscated']))
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val deid_pipeline = PretrainedPipeline("clinical_deidentification","de","clinical/models")

val sample = "Zusammenfassung : Michael Berger wird am Morgen des 12 Dezember 2018 ins St.Elisabeth Krankenhaus eingeliefert. 
Herr Michael Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen.

Persönliche Daten :
ID-Nummer: T0110053F
Platte A-BC124
Kontonummer: DE89370400440532013000
SSN : 13110587M565
Lizenznummer: B072RRE2I55
Adresse : St.Johann-Straße 13 19300"

val result = deid_pipeline.annotate(sample)
```
</div>

## Results

```bash
Masked with entity labels
------------------------------
Zusammenfassung : <PATIENT> wird am Morgen des <DATE> ins <HOSPITAL> eingeliefert.
Herr <PATIENT> ist <AGE> Jahre alt und hat zu viel Wasser in den Beinen.
Persönliche Daten :
ID-Nummer: <ID>
Platte <PLATE>
Kontonummer: <ACCOUNT>
SSN : <SSN>
Lizenznummer: <DLN>
Adresse : <STREET> <ZIP>

Masked with chars
------------------------------
Zusammenfassung : [************] wird am Morgen des [**************] ins [**********************] eingeliefert.
Herr [************] ist ** Jahre alt und hat zu viel Wasser in den Beinen.
Persönliche Daten :
ID-Nummer: [*******]
Platte [*****]
Kontonummer: [********************]
SSN : [**********]
Lizenznummer: [*********]
Adresse : [*****************] [***]

Masked with fixed length chars
------------------------------
Zusammenfassung : **** wird am Morgen des **** ins **** eingeliefert.
Herr **** ist **** Jahre alt und hat zu viel Wasser in den Beinen.
Persönliche Daten :
ID-Nummer: ****
Platte ****
Kontonummer: ****
SSN : ****
Lizenznummer: ****
Adresse : **** ****

Obfusceted
------------------------------
Zusammenfassung : Herrmann Kallert wird am Morgen des 11-26-1977 ins International Neuroscience eingeliefert.
Herr Herrmann Kallert ist 79 Jahre alt und hat zu viel Wasser in den Beinen.
Persönliche Daten :
ID-Nummer: 136704D357
Platte QA348G
Kontonummer: 192837465738
SSN : 1310011981M454
Lizenznummer: XX123456
Adresse : Klingelhöferring 31206
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clinical_deidentification|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.0+|
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
- DeIdentificationModel
- Finisher
