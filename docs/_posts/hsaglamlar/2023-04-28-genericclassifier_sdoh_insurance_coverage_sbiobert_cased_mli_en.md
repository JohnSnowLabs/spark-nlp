---
layout: model
title: SDOH Insurance Coverage For Classification
author: John Snow Labs
name: genericclassifier_sdoh_insurance_coverage_sbiobert_cased_mli
date: 2023-04-28
tags: [insurance, en, licensed, sdoh, social_determinants, public_health, coverage, open_source]
task: Text Classification
language: en
edition: Spark NLP 4.4.0
spark_version: [3.2, 3.0]
supported: true
annotator: GenericClassifierModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This Generic Classifier model is intended for detecting insurance coverage. In this classifier, we know/assume that the patient **has insurance**. 


`Good`: The insurance covers all or most of the medications.

`Poor`: The insurance doesn't cover all medications, specialist visits, or prescription medications. That may affect the patient's treatment.

`Unknown`: Insurance coverage is not mentioned in the clinical notes or is not known.

## Predicted Entities

`Good`, `Poor`, `Unknown`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/genericclassifier_sdoh_insurance_coverage_sbiobert_cased_mli_en_4.4.0_3.2_1682709767082.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/genericclassifier_sdoh_insurance_coverage_sbiobert_cased_mli_en_4.4.0_3.2_1682709767082.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
        
sentence_embeddings = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", 'en','clinical/models')\
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")

features_asm = FeaturesAssembler()\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("features")

generic_classifier = GenericClassifierModel.pretrained("genericclassifier_sdoh_insurance_coverage_sbiobert_cased_mli", 'en', 'clinical/models')\
    .setInputCols(["features"])\
    .setOutputCol("prediction")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_embeddings,
    features_asm,
    generic_classifier    
])

text_list = ["The patient's Medicaid insurance is limited with some medicaitons.", 
"She is under good coverage Medicare insurance",
"The patient has poor coverage of Private insurance",
"""Medical File for John Smith, Male, Age 42

Chief Complaint: Patient complains of nausea, vomiting, and shortness of breath.

History of Present Illness: The patient has a history of hypertension and diabetes, which are both poorly controlled. The patient has been feeling unwell for the past week, with symptoms including nausea, vomiting, and shortness of breath. Upon examination, the patient was found to have a high serum creatinine level of 5.8 mg/dL, indicating renal failure.

Past Medical History: The patient has a history of hypertension and diabetes, which have been poorly controlled due to poor medication adherence. The patient also has a history of smoking, which has been a contributing factor to the development of renal failure.

Medications: The patient is currently taking Metformin and Lisinopril for the management of diabetes and hypertension, respectively. However, due to poor Medicaid coverage, the patient is unable to afford some of the medications prescribed by his physician.

Insurance Status: The patient has Medicaid insurance, which provides poor coverage for some of the medications needed to manage his medical conditions, including those related to his renal failure.

Physical Examination: Upon physical examination, the patient appears pale and lethargic. Blood pressure is 160/100 mmHg, heart rate is 90 beats per minute, and respiratory rate is 20 breaths per minute. There is diffuse abdominal tenderness on palpation, and lung auscultation reveals bilateral rales.

Diagnosis: The patient is diagnosed with acute renal failure, likely due to uncontrolled hypertension and poorly managed diabetes.

Treatment: The patient is started on intravenous fluids and insulin to manage his blood sugar levels. Due to the patient's poor Medicaid coverage, the physician works with the patient to identify alternative medications that are more affordable and will still provide effective management of his medical conditions.

Follow-Up: The patient is advised to follow up with his primary care physician for ongoing management of his renal failure and other medical conditions. The patient is also referred to a nephrologist for further evaluation and management of his renal failure.
""",

"""Certainly, here is an example case study for a patient with private insurance:

Case Study for Emily Chen, Female, Age 38

Chief Complaint: Patient reports chronic joint pain and stiffness.

History of Present Illness: The patient has been experiencing chronic joint pain and stiffness, particularly in the hands, knees, and ankles. The pain is worse in the morning and improves throughout the day. The patient has also noticed some swelling and redness in the affected joints.

Past Medical History: The patient has a history of osteoarthritis, which has been gradually worsening over the past several years. The patient has tried over-the-counter pain relievers and joint supplements, but has not found significant relief.

Medications: The patient is currently taking over-the-counter pain relievers and joint supplements for the management of her osteoarthritis.

Insurance Status: The patient has private insurance, which provides comprehensive coverage for her medical care, including specialist visits and prescription medications.

Physical Examination: Upon physical examination, the patient has tenderness and swelling in multiple joints, particularly the hands, knees, and ankles. Range of motion is limited due to pain and stiffness.

Diagnosis: The patient is diagnosed with osteoarthritis, a chronic degenerative joint disease that causes pain, swelling, and stiffness in the affected joints.

Treatment: The patient is prescribed a nonsteroidal anti-inflammatory drug (NSAID) to manage pain and inflammation. The physician also recommends physical therapy to improve range of motion and strengthen the affected joints. The patient is advised to continue taking joint supplements for ongoing joint health.

Follow-Up: The patient is advised to follow up with the physician in 4-6 weeks to monitor response to treatment and make any necessary adjustments. The patient is also referred to a rheumatologist for further evaluation and management of her osteoarthritis."""]

df = spark.createDataFrame(text_list, StringType()).toDF("text")

result = pipeline.fit(df).transform(df)

result.select("text", "prediction.result").show(truncate=100)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
        
val sentence_embeddings = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
    .setInputCols("document")
    .setOutputCol("sentence_embeddings")

val features_asm = new FeaturesAssembler()
    .setInputCols("sentence_embeddings")
    .setOutputCol("features")

val generic_classifier = GenericClassifierModel.pretrained("genericclassifier_sdoh_insurance_coverage_sbiobert_cased_mli", "en", "clinical/models")
    .setInputCols("features")
    .setOutputCol("prediction")

val pipeline = new PipelineModel().setStages(Array(
    document_assembler,
    sentence_embeddings,
    features_asm,
    generic_classifier))

val data = Seq(Array("The patient's Medicaid insurance is limited with some medicaitons.", 
"She is under good coverage Medicare insurance",
"The patient has poor coverage of Private insurance",
"""Medical File for John Smith, Male, Age 42

Chief Complaint: Patient complains of nausea, vomiting, and shortness of breath.

History of Present Illness: The patient has a history of hypertension and diabetes, which are both poorly controlled. The patient has been feeling unwell for the past week, with symptoms including nausea, vomiting, and shortness of breath. Upon examination, the patient was found to have a high serum creatinine level of 5.8 mg/dL, indicating renal failure.

Past Medical History: The patient has a history of hypertension and diabetes, which have been poorly controlled due to poor medication adherence. The patient also has a history of smoking, which has been a contributing factor to the development of renal failure.

Medications: The patient is currently taking Metformin and Lisinopril for the management of diabetes and hypertension, respectively. However, due to poor Medicaid coverage, the patient is unable to afford some of the medications prescribed by his physician.

Insurance Status: The patient has Medicaid insurance, which provides poor coverage for some of the medications needed to manage his medical conditions, including those related to his renal failure.

Physical Examination: Upon physical examination, the patient appears pale and lethargic. Blood pressure is 160/100 mmHg, heart rate is 90 beats per minute, and respiratory rate is 20 breaths per minute. There is diffuse abdominal tenderness on palpation, and lung auscultation reveals bilateral rales.

Diagnosis: The patient is diagnosed with acute renal failure, likely due to uncontrolled hypertension and poorly managed diabetes.

Treatment: The patient is started on intravenous fluids and insulin to manage his blood sugar levels. Due to the patient's poor Medicaid coverage, the physician works with the patient to identify alternative medications that are more affordable and will still provide effective management of his medical conditions.

Follow-Up: The patient is advised to follow up with his primary care physician for ongoing management of his renal failure and other medical conditions. The patient is also referred to a nephrologist for further evaluation and management of his renal failure.
""",

"""Certainly, here is an example case study for a patient with private insurance:

Case Study for Emily Chen, Female, Age 38

Chief Complaint: Patient reports chronic joint pain and stiffness.

History of Present Illness: The patient has been experiencing chronic joint pain and stiffness, particularly in the hands, knees, and ankles. The pain is worse in the morning and improves throughout the day. The patient has also noticed some swelling and redness in the affected joints.

Past Medical History: The patient has a history of osteoarthritis, which has been gradually worsening over the past several years. The patient has tried over-the-counter pain relievers and joint supplements, but has not found significant relief.

Medications: The patient is currently taking over-the-counter pain relievers and joint supplements for the management of her osteoarthritis.

Insurance Status: The patient has private insurance, which provides comprehensive coverage for her medical care, including specialist visits and prescription medications.

Physical Examination: Upon physical examination, the patient has tenderness and swelling in multiple joints, particularly the hands, knees, and ankles. Range of motion is limited due to pain and stiffness.

Diagnosis: The patient is diagnosed with osteoarthritis, a chronic degenerative joint disease that causes pain, swelling, and stiffness in the affected joints.

Treatment: The patient is prescribed a nonsteroidal anti-inflammatory drug (NSAID) to manage pain and inflammation. The physician also recommends physical therapy to improve range of motion and strengthen the affected joints. The patient is advised to continue taking joint supplements for ongoing joint health.

Follow-Up: The patient is advised to follow up with the physician in 4-6 weeks to monitor response to treatment and make any necessary adjustments. The patient is also referred to a rheumatologist for further evaluation and management of her osteoarthritis.""")).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+------+
|                                                                                                text|result|
+----------------------------------------------------------------------------------------------------+------+
|                                  The patient's Medicaid insurance is limited with some medicaitons.|[Poor]|
|                                                       She is under good coverage Medicare insurance|[Good]|
|                                                  The patient has poor coverage of Private insurance|[Poor]|
|Medical File for John Smith, Male, Age 42\n\nChief Complaint: Patient complains of nausea, vomiti...|[Poor]|
|Certainly, here is an example case study for a patient with private insurance:\n\nCase Study for ...|[Good]|
+----------------------------------------------------------------------------------------------------+------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|genericclassifier_sdoh_insurance_coverage_sbiobert_cased_mli|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[features]|
|Output Labels:|[prediction]|
|Language:|en|
|Size:|3.4 MB|
|Dependencies:|sbiobert_base_cased_mli|

## References

Internal SDOH Project

## Benchmarking

```bash
       label  precision    recall  f1-score   support
        Good       0.81      0.86      0.84        74
        Poor       0.94      0.84      0.89        70
     Unknown       0.67      0.71      0.69        31
    accuracy        -         -        0.83       175
   macro-avg       0.80      0.81      0.80       175
weighted-avg       0.84      0.83      0.83       175
```