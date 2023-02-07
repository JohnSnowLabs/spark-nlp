---
layout: model
title: Clinical Deidentification (English, Glove, Augmented)
author: John Snow Labs
name: clinical_deidentification_glove_augmented
date: 2022-06-30
tags: [en, deid, deidentification, clinical, licensed]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pipeline is trained with lightweight `glove_100d` embeddings and can be used to deidentify PHI information from medical texts. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask and obfuscate `AGE`, `CONTACT`, `DATE`, `ID`, `LOCATION`, `NAME`, `PROFESSION`, `CITY`, `COUNTRY`, `DOCTOR`, `HOSPITAL`, `IDNUM`, `MEDICALRECORD`, `ORGANIZATION`, `PATIENT`, `PHONE`, `PROFESSION`,  `STREET`, `USERNAME`, `ZIP`, `ACCOUNT`, `LICENSE`, `VIN`, `SSN`, `DLN`, `PLATE`, `IPADDR` entities.

It's different to `clinical_deidentification_glove` in the way it manages PHONE and PATIENT, having apart from the NER, some rules in Contextual Parser components.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT_MULTI/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/DEID_PHI_TEXT_MULTI.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_glove_augmented_en_4.0.0_3.0_1656579032191.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_glove_augmented_en_4.0.0_3.0_1656579032191.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

deid_pipeline = PretrainedPipeline("clinical_deidentification_glove_augmented", "en", "clinical/models")

deid_pipeline.annotate("""Record date : 2093-01-13, David Hale, M.D. IP: 203.120.223.13. The driver's license no:A334455B. the SSN: 324598674 and e-mail: hale@gmail.com. Name : Hendrickson, Ora MR. # 719435 Date : 01/13/93. PCP : Oliveira, 25 years old. Record date : 2079-11-09, Patient's VIN : 1HGBH41JXMN109286.""")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val deid_pipeline = PretrainedPipeline("clinical_deidentification_glove_augmented","en","clinical/models")

val result = pipeline.annotate("""Record date : 2093-01-13, David Hale, M.D. IP: 203.120.223.13. The driver's license no:A334455B. the SSN: 324598674 and e-mail: hale@gmail.com. Name : Hendrickson, Ora MR. # 719435 Date : 01/13/93. PCP : Oliveira, 25 years old. Record date : 2079-11-09, Patient's VIN : 1HGBH41JXMN109286.""")
```
</div>

## Results

```bash
{'masked': ['Record date : <DATE>, <DOCTOR>, M.D.',
    'IP: <IPADDR>.',
    "The driver's license no: <LICENSE>.",
    'The SSN: <SSN> and e-mail: <EMAIL>.',
    'Name : <PATIENT> MR. # <MEDICALRECORD> Date : <DATE>.',
    'PCP : <DOCTOR>, <AGE> years old.',
    'Record date : <DATE>, <DOCTOR> : <VIN>.'],
 'masked_fixed_length_chars': ['Record date : ****, ****, M.D.',
    'IP: ****.',
    "The driver's license no: ****.",
    'The SSN: **** and e-mail: ****.',
    'Name : **** MR. # **** Date : ****.',
    'PCP : ****, **** years old.',
    'Record date : ****, **** : ****.'],
 'masked_with_chars': ['Record date : [********], [********], M.D.',
    'IP: [************].',
    "The driver's license no: [******].",
    'The SSN: [*******] and e-mail: [************].',
    'Name : [**************] MR. # [****] Date : [******].',
    'PCP : [******], ** years old.',
    'Record date : [********], [***********] : [***************].'],
 'ner_chunk': ['2093-01-13',
    'David Hale',
    'A334455B',
    '324598674',
    'hale@gmail.com',
    'Hendrickson, Ora',
    '719435',
    '01/13/93',
    'Oliveira',
    '25',
    '2079-11-09',
    "Patient's VIN",
    '1HGBH41JXMN109286'],
 'obfuscated': ['Record date : 2093-01-23, Dr Marshia Curling, M.D.',
    'IP: 004.004.004.004.',
    "The driver's license no: 123XX123.",
    'The SSN: SSN-089-89-9294 and e-mail: Mikey@hotmail.com.',
    'Name : Stephania Chang MR. # E5881795 Date : 02-14-1983.',
    'PCP : Dr Lovella Israel, 52 years old.',
    'Record date : 2079-11-14, Dr Colie Carne : 3CCCC22DDDD333888.'],
 'sentence': ['Record date : 2093-01-13, David Hale, M.D.',
    'IP: 203.120.223.13.',
    "The driver's license no: A334455B.",
    'The SSN: 324598674 and e-mail: hale@gmail.com.',
    'Name : Hendrickson, Ora MR. # 719435 Date : 01/13/93.',
    'PCP : Oliveira, 25 years old.',
    "Record date : 2079-11-09, Patient's VIN : 1HGBH41JXMN109286."]}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clinical_deidentification_glove_augmented|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|182.7 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel
- MedicalNerModel
- NerConverterInternalModel
- ChunkMergeModel
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- ChunkMergeModel
- ChunkMergeModel
- DeIdentificationModel
- DeIdentificationModel
- DeIdentificationModel
- DeIdentificationModel
- Finisher
