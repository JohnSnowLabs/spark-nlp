---
layout: model
title: Clinical Deidentification Pipeline (English, slim)
author: John Snow Labs
name: clinical_deidentification_slim
date: 2022-07-24
tags: [deidentification, deid, glove, slim, pipeline, clinical, en, licensed]
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

This pipeline is trained with lightweight `glove_100d` embeddings and can be used to deidentify PHI information from medical texts. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask and obfuscate `AGE`, `CONTACT`, `DATE`, `ID`, `LOCATION`, `NAME`, `PROFESSION`, `CITY`, `COUNTRY`, `DOCTOR`, `HOSPITAL`, `IDNUM`, `MEDICALRECORD`, `ORGANIZATION`, `PATIENT`, `PHONE`, `PROFESSION`, `STREET`, `USERNAME`, `ZIP`, `ACCOUNT`, `LICENSE`, `VIN`, `SSN`, `DLN`, `PLATE`, `IPADDR`, `EMAIL` entities.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_slim_en_4.0.0_3.0_1658699267236.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_slim_en_4.0.0_3.0_1658699267236.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

deid_pipeline = PretrainedPipeline("clinical_deidentification_slim", "en", "clinical/models")

sample = """Name : Hendrickson, Ora, Record date: 2093-01-13, # 719435.
Dr. John Green, ID: 1231511863, IP 203.120.223.13.
He is a 60-year-old male was admitted to the Day Hospital for cystectomy on 01/13/93.
Patient's VIN : 1HGBH41JXMN109286, SSN #333-44-6666, Driver's license no:A334455B.
Phone (302) 786-5227, 0295 Keats Street, San Francisco, E-MAIL: smith@gmail.com."""

result = deid_pipeline.annotate(sample)
print("\n".join(result['masked']))
print("\n".join(result['masked_with_chars']))
print("\n".join(result['masked_fixed_length_chars']))
print("\n".join(result['obfuscated']))
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val deid_pipeline = new PretrainedPipeline("clinical_deidentification_slim","en","clinical/models")

val sample = """Name : Hendrickson, Ora, Record date: 2093-01-13, # 719435.
Dr. John Green, ID: 1231511863, IP 203.120.223.13.
He is a 60-year-old male was admitted to the Day Hospital for cystectomy on 01/13/93.
Patient's VIN : 1HGBH41JXMN109286, SSN #333-44-6666, Driver's license no:A334455B.
Phone (302) 786-5227, 0295 Keats Street, San Francisco, E-MAIL: smith@gmail.com."""

val result = deid_pipeline.annotate(sample)
```
</div>

## Results

```bash
Masked with entity labels
------------------------------
Name : <PATIENT>, Record date: <DATE>, # <MEDICALRECORD>.
Dr. <DOCTOR>, ID: <IDNUM>, IP <IPADDR>.
He is a <AGE> male was admitted to the <HOSPITAL> for cystectomy on <DATE>.
Patient's VIN : <VIN>, SSN <SSN>, Driver's license <DLN>.
Phone <PHONE>, <STREET>, <CITY>, E-MAIL: <EMAIL>.

Masked with chars
------------------------------
Name : [**************], Record date: [********], # [****].
Dr. [********], ID: [********], IP [************].
He is a [*********] male was admitted to the [**********] for cystectomy on [******].
Patient's VIN : [***************], SSN [**********], Driver's license [*********].
Phone [************], [***************], [***********], E-MAIL: [*************].

Masked with fixed length chars
------------------------------
Name : ****, Record date: ****, # ****.
Dr. ****, ID: ****, IP ****.
He is a **** male was admitted to the **** for cystectomy on ****.
Patient's VIN : ****, SSN ****, Driver's license ****.
Phone ****, ****, ****, E-MAIL: ****.

Obfuscated
------------------------------
Name : Layne Nation, Record date: 2093-03-13, # C6240488.
Dr. Dr Rosalba Hill, ID: JY:3489547, IP 005.005.005.005.
He is a 79 male was admitted to the JOHN MUIR MEDICAL CENTER-CONCORD CAMPUS for cystectomy on 01-25-1997.
Patient's VIN : 3CCCC22DDDD333888, SSN SSN-289-37-4495, Driver's license S99983662.
Phone 04.32.52.27.90, North Adrienne, Colorado Springs, E-MAIL: Rawland@google.com.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clinical_deidentification_slim|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|181.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
- MedicalNerModel
- NerConverter
- ChunkMergeModel
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- TextMatcherModel
- ContextualParserModel
- RegexMatcherModel
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