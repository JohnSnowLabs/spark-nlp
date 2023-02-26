---
layout: model
title: Voice of the Patients
author: John Snow Labs
name: ner_vop_slim_wip
date: 2023-02-25
tags: [ner, clinical, en, licensed, vop, voice, patient]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.3.1
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts healthcare-related terms from the documents transferred from the patient's own sentences.

Note: 'wip' suffix indicates that the model development is work-in-progress and will be finalised and the model performance will improved in the upcoming releases.

## Predicted Entities

`AdmissionDischarge`, `Age`, `BodyPart`, `ClinicalDept`, `DateTime`, `Disease`, `Dosage_Strength`, `Drug`, `Duration`, `Employment`, `Form`, `Frequency`, `Gender`, `Laterality`, `Procedure`, `PsychologicalCondition`, `RelationshipStatus`, `Route`, `Symptom`, `Test`, `Vaccine`, `VitalTest`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_vop_slim_wip_en_4.3.1_3.0_1677342424243.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_vop_slim_wip_en_4.3.1_3.0_1677342424243.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

clinical_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_vop_slim_wip", "en", "clinical/models")\
    .setInputCols(["sentence", "token","embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler, 
    sentence_detector,
    tokenizer,
    clinical_embeddings,
    ner_model,
    ner_converter   
    ])

sample_texts = ["Hello,I'm 20 year old girl. I'm diagnosed with hyperthyroid 1 month ago. I was feeling weak, light headed,poor digestion, panic attacks, depression, left chest pain, increased heart rate, rapidly weight loss,  from 4 months. Because of this, I stayed in the hospital and just discharged from hospital. I had many other blood tests, brain mri, ultrasound scan, endoscopy because of some dumb doctors bcs they were not able to diagnose actual problem. Finally I got an appointment with a homeopathy doctor finally he find that i was suffering from hyperthyroid and my TSH was 0.15 T3 and T4 is normal . Also i have b12 deficiency and vitamin D deficiency so I'm taking weekly supplement of vitamin D and 1000 mcg b12 daily. I'm taking homeopathy medicine for 40 days and took 2nd test after 30 days. My TSH is 0.5 now. I feel a little bit relief from weakness and depression but I'm facing with 2 new problem from last week that is breathtaking problem and very rapid heartrate. I just want to know if i should start allopathy medicine or homeopathy is okay? Bcs i heard that thyroid take time to start recover. So please let me know if both of medicines take same time. Because some of my friends advising me to start allopathy and never take a chance as i can develop some serious problems.Sorry for my poor english😐Thank you."]


data = spark.createDataFrame(sample_texts, StringType()).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val clinical_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_vop_slim_wip", "en", "clinical/models")
    .setInputCols(Array("sentence", "token","embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(
    document_assembler, 
    sentence_detector,
    tokenizer,
    clinical_embeddings,
    ner_model,
    ner_converter   
))

val data = Seq("Hello,I'm 20 year old girl. I'm diagnosed with hyperthyroid 1 month ago. I was feeling weak, light headed,poor digestion, panic attacks, depression, left chest pain, increased heart rate, rapidly weight loss,  from 4 months. Because of this, I stayed in the hospital and just discharged from hospital. I had many other blood tests, brain mri, ultrasound scan, endoscopy because of some dumb doctors bcs they were not able to diagnose actual problem. Finally I got an appointment with a homeopathy doctor finally he find that i was suffering from hyperthyroid and my TSH was 0.15 T3 and T4 is normal . Also i have b12 deficiency and vitamin D deficiency so I'm taking weekly supplement of vitamin D and 1000 mcg b12 daily. I'm taking homeopathy medicine for 40 days and took 2nd test after 30 days. My TSH is 0.5 now. I feel a little bit relief from weakness and depression but I'm facing with 2 new problem from last week that is breathtaking problem and very rapid heartrate. I just want to know if i should start allopathy medicine or homeopathy is okay? Bcs i heard that thyroid take time to start recover. So please let me know if both of medicines take same time. Because some of my friends advising me to start allopathy and never take a chance as i can develop some serious problems.Sorry for my poor english😐Thank you.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+--------------------+-----+----+----------------------+
|chunk               |begin|end |ner_label             |
+--------------------+-----+----+----------------------+
|20 year old         |10   |20  |Age                   |
|girl                |22   |25  |Gender                |
|hyperthyroid        |47   |58  |Disease               |
|1 month ago         |60   |70  |DateTime              |
|weak                |87   |90  |Symptom               |
|panic attacks       |122  |134 |PsychologicalCondition|
|depression          |137  |146 |PsychologicalCondition|
|left                |149  |152 |Laterality            |
|chest               |154  |158 |BodyPart              |
|pain                |160  |163 |Symptom               |
|heart rate          |176  |185 |VitalTest             |
|weight loss         |196  |206 |Symptom               |
|4 months            |215  |222 |Duration              |
|hospital            |258  |265 |ClinicalDept          |
|discharged          |276  |285 |AdmissionDischarge    |
|hospital            |292  |299 |ClinicalDept          |
|blood tests         |319  |329 |Test                  |
|brain               |332  |336 |BodyPart              |
|mri                 |338  |340 |Test                  |
|ultrasound scan     |343  |357 |Test                  |
|endoscopy           |360  |368 |Procedure             |
|doctors             |391  |397 |Employment            |
|homeopathy doctor   |486  |502 |Employment            |
|he                  |512  |513 |Gender                |
|hyperthyroid        |546  |557 |Disease               |
|TSH                 |566  |568 |Test                  |
|T3                  |579  |580 |Test                  |
|T4                  |586  |587 |Test                  |
|b12 deficiency      |613  |626 |Disease               |
|vitamin D deficiency|632  |651 |Disease               |
|weekly              |667  |672 |Frequency             |
|supplement          |674  |683 |Drug                  |
|vitamin D           |688  |696 |Drug                  |
|1000 mcg            |702  |709 |Dosage_Strength       |
|b12                 |711  |713 |Drug                  |
|daily               |715  |719 |Frequency             |
|homeopathy medicine |733  |751 |Drug                  |
|40 days             |757  |763 |Duration              |
|after 30 days       |783  |795 |DateTime              |
|TSH                 |801  |803 |Test                  |
|now                 |812  |814 |DateTime              |
|weakness            |849  |856 |Symptom               |
|depression          |862  |871 |PsychologicalCondition|
|last week           |912  |920 |DateTime              |
|rapid heartrate     |960  |974 |Symptom               |
|thyroid             |1074 |1080|BodyPart              |
+--------------------+-----+----+----------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_vop_slim_wip|
|Compatibility:|Healthcare NLP 4.3.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|2.4 MB|

## Benchmarking

```bash
               label     tp    fp    fn  total precision recall     f1
               Route   25.0   4.0  15.0   40.0    0.8621  0.625 0.7246
           Procedure  161.0  48.0  70.0  231.0    0.7703  0.697 0.7318
             Vaccine   22.0   7.0   8.0   30.0    0.7586 0.7333 0.7458
  RelationshipStatus    6.0   2.0   2.0    8.0      0.75   0.75   0.75
             Disease  884.0 201.0 285.0 1169.0    0.8147 0.7562 0.7844
           Frequency  342.0  61.0 113.0  455.0    0.8486 0.7516 0.7972
            Duration  720.0 188.0 146.0  866.0     0.793 0.8314 0.8117
                Test  478.0 106.0 103.0  581.0    0.8185 0.8227 0.8206
             Symptom 1569.0 337.0 340.0 1909.0    0.8232 0.8219 0.8225
            DateTime 1558.0 277.0 296.0 1854.0     0.849 0.8403 0.8447
        ClinicalDept  157.0   9.0  48.0  205.0    0.9458 0.7659 0.8464
                Form  110.0  28.0  11.0  121.0    0.7971 0.9091 0.8494
     Dosage_Strength  184.0  25.0  33.0  217.0    0.8804 0.8479 0.8638
                Drug  672.0 109.0 103.0  775.0    0.8604 0.8671 0.8638
           VitalTest   73.0   7.0  16.0   89.0    0.9125 0.8202 0.8639
          Laterality  262.0  43.0  38.0  300.0     0.859 0.8733 0.8661
                 Age  236.0  42.0  14.0  250.0    0.8489  0.944 0.8939
PsychologicalCond...  144.0  20.0  14.0  158.0     0.878 0.9114 0.8944
            BodyPart 1319.0 139.0 160.0 1479.0    0.9047 0.8918 0.8982
          Employment  541.0  25.0  77.0  618.0    0.9558 0.8754 0.9139
  AdmissionDischarge   13.0   0.0   1.0   14.0       1.0 0.9286  0.963
              Gender  548.0  26.0  12.0  560.0    0.9547 0.9786 0.9665
```
