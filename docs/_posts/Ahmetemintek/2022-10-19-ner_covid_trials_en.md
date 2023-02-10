---
layout: model
title: Extract Entities in Covid Trials
author: John Snow Labs
name: ner_covid_trials
date: 2022-10-19
tags: [ner, en, clinical, licensed, covid]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 4.2.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained named entity recognition deep learning model for extracting covid-related clinical terminology from covid trials.

## Predicted Entities

`Stage`, `Severity`, `Virus`, `Trial_Design`, `Trial_Phase`, `N_Patients`, `Institution`, `Statistical_Indicator`, `Section_Header`, `Cell_Type`, `Cellular_component`, `Viral_components`, `Physiological_reaction`, `Biological_molecules`, `Admission_Discharge`, `Age`, `BMI`, `Cerebrovascular_Disease`, `Date`, `Death_Entity`, `Diabetes`, `Disease_Syndrome_Disorder`, `Dosage`, `Drug_Ingredient`, `Employment`, `Frequency`, `Gender`, `Heart_Disease`, `Hypertension`, `Obesity`, `Pulse`, `Race_Ethnicity`, `Respiration`, `Route`, `Smoking`, `Time`, `Total_Cholesterol`, `Treatment`, `VS_Finding`, `Vaccine`, `Vaccine_Name`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_COVID/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_covid_trials_en_4.2.0_3.0_1666177383134.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_covid_trials_en_4.2.0_3.0_1666177383134.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") 

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner = MedicalNerModel.pretrained("ner_covid_trials","en","clinical/models")\
    .setInputCols(["sentence","token","embeddings"])\
    .setOutputCol("ner")\
    .setLabelCasing("upper")
    
ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")

ner_pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    ner,
    ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

ner_model = ner_pipeline.fit(empty_data)

text= """In December 2019 , a group of patients with the acute respiratory disease was detected in Wuhan , Hubei Province of China . A month later , a new beta-coronavirus was identified as the cause of the 2019 coronavirus infection . SARS-CoV-2 is a coronavirus that belongs to the group of β-coronaviruses of the subgenus Coronaviridae . The SARS-CoV-2 is the third known zoonotic coronavirus disease after severe acute respiratory syndrome ( SARS ) and Middle Eastern respiratory syndrome ( MERS ). The diagnosis of SARS-CoV-2 recommended by the WHO , CDC is the collection of a sample from the upper respiratory tract ( nasal and oropharyngeal exudate ) or from the lower respiratory tractsuch as expectoration of endotracheal aspirate and bronchioloalveolar lavage and its analysis using the test of real-time polymerase chain reaction ( qRT-PCR ).In 2020, the first COVID‑19 vaccine was developed and made available to the public through emergency authorizations and conditional approvals."""

results= model.transform(spark.createDataFrame([[text]]).toDF('text'))
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical" ,"en", "clinical/models")
    .setInputCols(Array("sentence","token"))
    .setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_covid_trials", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, 
                                            sentence_detector, 
                                            tokenizer, 
                                            word_embeddings, 
                                            ner_model, 
                                            ner_converter))

val data = Seq("""In December 2019 , a group of patients with the acute respiratory disease was detected in Wuhan , Hubei Province of China . A month later , a new beta-coronavirus was identified as the cause of the 2019 coronavirus infection . SARS-CoV-2 is a coronavirus that belongs to the group of β-coronaviruses of the subgenus Coronaviridae . The SARS-CoV-2 is the third known zoonotic coronavirus disease after severe acute respiratory syndrome ( SARS ) and Middle Eastern respiratory syndrome ( MERS ). The diagnosis of SARS-CoV-2 recommended by the WHO , CDC is the collection of a sample from the upper respiratory tract ( nasal and oropharyngeal exudate ) or from the lower respiratory tractsuch as expectoration of endotracheal aspirate and bronchioloalveolar lavage and its analysis using the test of real-time polymerase chain reaction ( qRT-PCR ).In 2020, the first COVID‑19 vaccine was developed and made available to the public through emergency authorizations and conditional approvals.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|    | chunks                              |   begin |   end |   sentence_id | entities                  |
|---:|:------------------------------------|--------:|------:|--------------:|:--------------------------|
|  0 | December 2019                       |       3 |    15 |             0 | Date                      |
|  1 | acute respiratory disease           |      48 |    72 |             0 | Disease_Syndrome_Disorder |
|  2 | beta-coronavirus                    |     146 |   161 |             1 | Virus                     |
|  3 | 2019                                |     198 |   201 |             1 | Date                      |
|  4 | coronavirus infection               |     203 |   223 |             1 | Disease_Syndrome_Disorder |
|  5 | SARS-CoV-2                          |     228 |   237 |             2 | Virus                     |
|  6 | coronavirus                         |     244 |   254 |             2 | Virus                     |
|  7 | β-coronaviruses                     |     285 |   299 |             2 | Virus                     |
|  8 | subgenus Coronaviridae              |     308 |   329 |             2 | Virus                     |
|  9 | SARS-CoV-2                          |     337 |   346 |             3 | Virus                     |
| 10 | zoonotic coronavirus disease        |     367 |   394 |             3 | Disease_Syndrome_Disorder |
| 11 | severe acute respiratory syndrome   |     402 |   434 |             3 | Disease_Syndrome_Disorder |
| 12 | SARS                                |     438 |   441 |             3 | Disease_Syndrome_Disorder |
| 13 | Middle Eastern respiratory syndrome |     449 |   483 |             3 | Disease_Syndrome_Disorder |
| 14 | MERS                                |     487 |   490 |             3 | Disease_Syndrome_Disorder |
| 15 | SARS-CoV-2                          |     513 |   522 |             4 | Virus                     |
| 16 | WHO                                 |     543 |   545 |             4 | Institution               |
| 17 | CDC                                 |     549 |   551 |             4 | Institution               |
| 18 | 2020                                |     852 |   855 |             5 | Date                      |
| 19 | COVID‑19 vaccine                    |     868 |   883 |             5 | Vaccine_Name              |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_covid_trials|
|Compatibility:|Spark NLP for Healthcare 4.2.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|14.8 MB|

## References

This model is trained on data sampled from clinicaltrials.gov - covid trials, and annotated in-house.

## Benchmarking

```bash
           label         tp     fp    fn   total  precision  recall  f1
         Institution     34      8    20  55.0     0.7958  0.6343   0.706
          VS_Finding     19      2     1  20.0     0.9048    0.95  0.9268
         Respiration      5      0     0   5.0        1.0     1.0     1.0
Cerebrovascular_D...      5      2     2   7.0     0.7143  0.7143  0.7143
           Cell_Type    152     27    14 167.0     0.8479  0.9123  0.8789
       Heart_Disease     36      3     5  41.0     0.9231   0.878     0.9
            Severity     57     25     3  60.0     0.6881    0.95  0.7981
          N_Patients     27      3     1  29.0     0.8871  0.9483  0.9167
               Pulse     12      2     0  12.0     0.8571     1.0  0.9231
             Obesity      3      0     0   3.0        1.0     1.0     1.0
 Admission_Discharge     85      3     0  85.0     0.9659     1.0  0.9827
            Diabetes      8      0     0   8.0        1.0     1.0     1.0
      Section_Header     94      8    13 108.0     0.9154  0.8711  0.8927
                 Age     22      1     0  22.0     0.9429     1.0  0.9706
  Cellular_component     40     21    10  50.0     0.6534     0.8  0.7193
        Hypertension     10      0     0  10.0        1.0     1.0     1.0
                 BMI      5      1     1   6.0     0.8333  0.8333  0.8333
         Trial_Phase     13      0     1  14.0     0.9398  0.9286  0.9341
          Employment     98     12     8 107.0     0.8874  0.9206  0.9037
Statistical_Indic...     76     29    11  88.0     0.7206  0.8689  0.7879
                Time      2      0     1   3.0        1.0  0.6667     0.8
   Total_Cholesterol     14      1     2  17.0     0.9355  0.8529  0.8923
     Drug_Ingredient    327     33    67 395.0     0.9084  0.8281  0.8664
Physiological_rea...     27      7    14  41.0     0.7864  0.6585  0.7168
           Treatment     66      4    25  92.0     0.9433  0.7228  0.8185
             Vaccine     20      1     2  23.0     0.9531  0.8841  0.9173
Disease_Syndrome_...    774     70    41 816.0     0.9171  0.9495   0.933
               Virus    121      8    23 144.0     0.9365  0.8403  0.8858
           Frequency     57      1     2  59.9     0.9787  0.9556   0.967
               Route     37      4    10  47.0     0.9024  0.7872  0.8409
        Death_Entity     20      9     3  23.0     0.6897  0.8696  0.7692
               Stage      4      0     7  12.0        1.0  0.3889    0.56
        Vaccine_Name     10      1     0  10.0     0.9091     1.0  0.9524
        Trial_Design     32     13     8  41.0     0.7149  0.7951  0.7529
Biological_molecules    251     91    53 305.0     0.7335  0.8233  0.7758
                Date     98      5     2 100.0     0.9492    0.98  0.9643
      Race_Ethnicity      0      0     2   2.0        0.0     0.0     0.0
              Gender     46      1     0  46.0     0.9787     1.0  0.9892
              Dosage     49      9    24  73.0     0.8376  0.6712  0.7452
    Viral_components     18     10    15  34.0     0.6512   0.549  0.5957

macro                    -     -    -     -        -        -      0.8382
micro                    -     -    -     -        -        -      0.8704
```