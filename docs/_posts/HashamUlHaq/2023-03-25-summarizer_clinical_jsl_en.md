---
layout: model
title: Summarize clinical notes
author: John Snow Labs
name: summarizer_clinical_jsl
date: 2023-03-25
tags: [en, licensed, clinical, summarization, tensorflow]
task: Summarization
language: en
edition: Healthcare NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: MedicalSummarizer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Summarize clinical notes, encounters, critical care notes, discharge notes, reports, etc.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/summarizer_clinical_jsl_en_4.3.1_3.0_1679772340755.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/summarizer_clinical_jsl_en_4.3.1_3.0_1679772340755.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document = sparknlp.annotators.DocumentAssembler().setInputCol('text').setOutputCol('document')

summarizer = sparknlp_jsl.annotators.MedicalSummarizer\
    .pretrained("summarizer_clinical_jsl").setInputCols(['document'])\
    .setOutputCol('summary')\
    .setMaxTextLength(512)\
    .setMaxNewTokens(512)

pipeline = sparknlp.base.Pipeline(stages=[
    document,
    summarizer  
])

text = ""'Patient with hypertension, syncope, and spinal stenosis - for recheck.
(Medical Transcription Sample Report)
SUBJECTIVE:
The patient is a 78-year-old female who returns for recheck. She has hypertension. She denies difficulty with chest pain, palpations, orthopnea, nocturnal dyspnea, or edema.
PAST MEDICAL HISTORY / SURGERY / HOSPITALIZATIONS:
Reviewed and unchanged from the dictation on 12/03/2003.
MEDICATIONS:
Atenolol 50 mg daily, Premarin 0.625 mg daily, calcium with vitamin D two to three pills daily, multivitamin daily, aspirin as needed, and TriViFlor 25 mg two pills daily. She also has Elocon cream 0.1% and Synalar cream 0.01% that she uses as needed for rash.'""

data = spark.createDataFrame([[text]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val summarizer = MedicalSummarizer
    .pretrained("summarizer_clinical_jsl")
    .setInputCols(['document'])
    .setOutputCol('summary')
    .setMaxTextLength(512)
    .setMaxNewTokens(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, summarizer))
```
</div>

## Results

```bash
A 78-year-old female with hypertension, syncope, and spinal stenosis returns for recheck. She denies chest pain, palpations, orthopnea, nocturnal dyspnea, or edema. She is on multiple medications and has Elocon cream and Synalar cream for rash.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|summarizer_clinical_jsl|
|Compatibility:|Healthcare NLP 4.3.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|920.1 MB|

## References

Trained on in-house curated dataset