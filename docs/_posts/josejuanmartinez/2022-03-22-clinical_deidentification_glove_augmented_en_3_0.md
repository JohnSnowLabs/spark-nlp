---
layout: model
title: Clinical Deidentification (English, Glove, Augmented)
author: John Snow Labs
name: clinical_deidentification_glove_augmented
date: 2022-03-22
tags: [deid, deidentification, en, licensed]
task: De-identification
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
recommended: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pipeline is trained with lightweight glove_100d embeddings and can be used to deidentify PHI information from medical texts. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask and obfuscate `AGE`, `CONTACT`, `DATE`, `ID`, `LOCATION`, `NAME`, `PROFESSION`, `CITY`, `COUNTRY`, `DOCTOR`, `HOSPITAL`, `IDNUM`, `MEDICALRECORD`, `ORGANIZATION`, `PATIENT`, `PHONE`, `PROFESSION`,  `STREET`, `USERNAME`, `ZIP`, `ACCOUNT`, `LICENSE`, `VIN`, `SSN`, `DLN`, `PLATE`, `IPADDR` entities.

It's different to `clinical_deidentification_glove` in the way it manages PHONE and PATIENT, having apart from the NER, some rules in Contextual Parser components.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_glove_augmented_en_3.4.1_3.0_1647966639326.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_glove_augmented_en_3.4.1_3.0_1647966639326.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from johnsnowlabs import *
deid_pipeline = PretrainedPipeline("clinical_deidentification_glove", "en", "clinical/models")

deid_pipeline.annotate("Record date : 2093-01-13, David Hale, M.D. IP: 203.120.223.13. The driver's license no:A334455B. the SSN:324598674 and e-mail: hale@gmail.com. Name : Hendrickson, Ora MR. # 719435 Date : 01/13/93. PCP : Oliveira, 25 years-old. Record date : 2079-11-09, Patient's VIN : 1HGBH41JXMN109286.")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val deid_pipeline = PretrainedPipeline("clinical_deidentification_glove","en","clinical/models")

val result = pipeline.annotate("Record date : 2093-01-13, David Hale, M.D. IP: 203.120.223.13. The driver's license no:A334455B. the SSN:324598674 and e-mail: hale@gmail.com. Name : Hendrickson, Ora MR. # 719435 Date : 01/13/93. PCP : Oliveira, 25 years-old. Record date : 2079-11-09, Patient's VIN : 1HGBH41JXMN109286.")
```
</div>

## Results

```bash
{'sentence': ['Record date : 2093-01-13, David Hale, M.D.',
   'IP: 203.120.223.13.',
   'The driver's license no:A334455B.',
   'the SSN:324598674 and e-mail: hale@gmail.com.',
   'Name : Hendrickson, Ora MR. # 719435 Date : 01/13/93.',
   'PCP : Oliveira, 25 years-old.',
   'Record date : 2079-11-09, Patient's VIN : 1HGBH41JXMN109286.'],
'masked': ['Record date : <DATE>, <DOCTOR>, M.D.',
   'IP: <IPADDR>.',
   'The driver's license <DLN>.',
   'the <SSN> and e-mail: <EMAIL>.',
   'Name : <PATIENT> MR. # <MEDICALRECORD> Date : <DATE>.',
   'PCP : <DOCTOR>, <AGE> years-old.',
   'Record date : <DATE>, Patient's VIN : <VIN>.'],
'obfuscated': ['Record date : 2093-02-13, Shella Solan, M.D.',
   'IP: 444.444.444.444.',
   'The driver's license O497302436569.',
   'the SSN-539-29-1060 and e-mail: Keith@google.com.',
   'Name : Roscoe Kerns MR. # Q984288 Date : 10-08-1991.',
   'PCP : Dr Rudell Dubin, 10 years-old.',
   'Record date : 2079-12-30, Patient's VIN : 5eeee44ffff555666.'],
'ner_chunk': ['2093-01-13',
   'David Hale',
   'no:A334455B',
   'SSN:324598674',
   'Hendrickson, Ora',
   '719435',
   '01/13/93',
   'Oliveira',
   '25',
   '2079-11-09',
   '1HGBH41JXMN109286']}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clinical_deidentification_glove_augmented|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|181.3 MB|

## Included Models

- nlp.DocumentAssembler
- nlp.SentenceDetector
- nlp.TokenizerModel
- nlp.WordEmbeddingsModel
- medical.NerModel
- medical.NerConverterInternal
- medical.NerModel
- medical.NerConverterInternal
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
- medical.DeIdentificationModel
- medical.DeIdentificationModel
- medical.DeIdentificationModel
- medical.DeIdentificationModel
- Finisher
