---
layout: model
title: Extract Cancer Therapies and Granular Posology Information
author: John Snow Labs
name: ner_oncology_posology_wip
date: 2022-10-01
tags: [licensed, clinical, oncology, en, ner, treatment, posology]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts cancer therapies (Cancer_Surgery, Radiotherapy and Cancer_Therapy) and posology information at a granular level.

Definitions of Predicted Entities:
- `Cancer_Surgery`: Terms that indicate surgery as a form of cancer treatment.
- `Cancer_Therapy`: Any cancer treatment mentioned in text, excluding surgeries and radiotherapy.
- `Cycle_Count`: The total number of cycles being administered of an oncological therapy (e.g. "5 cycles"). 
- `Cycle_Day`: References to the day of the cycle of oncological therapy (e.g. "day 5").
- `Cycle_Number`: The number of the cycle of an oncological therapy that is being applied (e.g. "third cycle").
- `Dosage`: The quantity prescribed by the physician for an active ingredient.
- `Duration`: Words indicating the duration of a treatment (e.g. "for 2 weeks").
- `Frequency`: Words indicating the frequency of treatment administration (e.g. "daily" or "bid").
- `Radiotherapy`: Terms that indicate the use of Radiotherapy.
- `Radiation_Dose`: Dose used in radiotherapy.
- `Route`: Words indicating the type of administration route (such as "PO" or "transdermal").



## Predicted Entities

`Cancer_Surgery`, `Cancer_Therapy`, `Cycle_Count`, `Cycle_Day`, `Cycle_Number`, `Dosage`, `Duration`, `Frequency`, `Radiotherapy`, `Radiation_Dose`, `Route`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_posology_wip_en_4.0.0_3.0_1664599604423.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")                

ner = MedicalNerModel.pretrained("ner_oncology_posology_wip", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")
pipeline = Pipeline(stages=[document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter])

data = spark.createDataFrame([["The patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses. She is currently receiving his second cycle of chemotherapy and is in good overall condition."]]).toDF("text")

result = pipeline.fit(data).transform(data)

```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    
val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
    .setInputCols(Array("document"))
    .setOutputCol("sentence")
    
val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")
    
val word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")                
    
val ner = MedicalNerModel.pretrained("ner_oncology_posology_wip", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")
    
val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

        
val pipeline = new Pipeline().setStages(Array(document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter))    

val data = Seq("The patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses. She is currently receiving his second cycle of chemotherapy and is in good overall condition.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)


```
</div>

## Results

```bash
| chunk            | ner_label      |
|:-----------------|:---------------|
| adriamycin       | Cancer_Therapy |
| 60 mg/m2         | Dosage         |
| cyclophosphamide | Cancer_Therapy |
| 600 mg/m2        | Dosage         |
| six courses      | Cycle_Count    |
| second cycle     | Cycle_Number   |
| chemotherapy     | Cancer_Therapy |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_posology_wip|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|856.4 KB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
         label     tp    fp    fn  total  precision  recall   f1
  Cycle_Number   56.0  12.0  17.0   73.0       0.82    0.77 0.79
   Cycle_Count  148.0  44.0  27.0  175.0       0.77    0.85 0.81
  Radiotherapy  185.0   2.0  37.0  222.0       0.99    0.83 0.90
Cancer_Surgery  494.0  59.0 151.0  645.0       0.89    0.77 0.82
     Cycle_Day  144.0  22.0  39.0  183.0       0.87    0.79 0.83
     Frequency  270.0  17.0  79.0  349.0       0.94    0.77 0.85
         Route   67.0   5.0  30.0   97.0       0.93    0.69 0.79
Cancer_Therapy 1093.0  74.0 165.0 1258.0       0.94    0.87 0.90
      Duration  316.0  43.0 231.0  547.0       0.88    0.58 0.70
        Dosage  703.0  16.0 124.0  827.0       0.98    0.85 0.91
Radiation_Dose   84.0  14.0  12.0   96.0       0.86    0.88 0.87
     macro_avg 3560.0 308.0 912.0 4472.0       0.90    0.78 0.83
     micro_avg    NaN   NaN   NaN    NaN       0.92    0.80 0.85
```