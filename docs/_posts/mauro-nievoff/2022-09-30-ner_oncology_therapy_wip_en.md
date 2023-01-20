---
layout: model
title: Detect Entities Related to Cancer Therapies
author: John Snow Labs
name: ner_oncology_therapy_wip
date: 2022-09-30
tags: [licensed, clinical, oncology, en, ner, treatment]
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

This model extracts entities related to oncology therapies using granular labels, including mentions of treatments, posology information and line of therapy.

Definitions of Predicted Entities:

- `Cancer_Surgery`: Terms that indicate surgery as a form of cancer treatment.
- `Chemotherapy`: Mentions of chemotherapy drugs, or unspecific words such as "chemotherapy".
- `Cycle_Count`: The total number of cycles being administered of an oncological therapy (e.g. "5 cycles"). 
- `Cycle_Day`: References to the day of the cycle of oncological therapy (e.g. "day 5").
- `Cycle_Number`: The number of the cycle of an oncological therapy that is being applied (e.g. "third cycle").
- `Dosage`: The quantity prescribed by the physician for an active ingredient.
- `Duration`: Words indicating the duration of a treatment (e.g. "for 2 weeks").
- `Frequency`: Words indicating the frequency of treatment administration (e.g. "daily" or "bid").
- `Hormonal_Therapy`: Mentions of hormonal drugs used to treat cancer, or unspecific words such as "hormonal therapy".
- `Immunotherapy`: Mentions of immunotherapy drugs, or unspecific words such as "immunotherapy".
- `Line_Of_Therapy`: Explicit references to the line of therapy of an oncological therapy (e.g. "first-line treatment").
- `Radiotherapy`: Terms that indicate the use of Radiotherapy.
- `Radiation_Dose`: Dose used in radiotherapy.
- `Response_To_Treatment`: Terms related to clinical progress of the patient related to cancer treatment, including "recurrence", "bad response" or "improvement".
- `Route`: Words indicating the type of administration route (such as "PO" or "transdermal").
- `Targeted_Therapy`: Mentions of targeted therapy drugs, or unspecific words such as "targeted therapy".
- `Unspecific_Therapy`: Terms that indicate a known cancer therapy but that is not specific to any other therapy entity (e.g. "chemoradiotherapy" or "adjuvant therapy").

## Predicted Entities

`Cancer_Surgery`, `Chemotherapy`, `Cycle_Count`, `Cycle_Day`, `Cycle_Number`, `Dosage`, `Duration`, `Frequency`, `Hormonal_Therapy`, `Immunotherapy`, `Line_Of_Therapy`, `Radiotherapy`, `Radiation_Dose`, `Response_To_Treatment`, `Route`, `Targeted_Therapy`, `Unspecific_Therapy` 



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_therapy_wip_en_4.0.0_3.0_1664557936894.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_therapy_wip_en_4.0.0_3.0_1664557936894.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner = MedicalNerModel.pretrained("ner_oncology_therapy_wip", "en", "clinical/models") \
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

data = spark.createDataFrame([["The had previously undergone a left mastectomy and an axillary lymph node dissection for a left breast cancer twenty years ago.The tumor was positive for ER and PR. Postoperatively, radiotherapy was administered to her breast.The cancer recurred as a right lung metastasis 13 years later. The patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses, as first line therapy."]]).toDF("text")

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
    
val ner = MedicalNerModel.pretrained("ner_oncology_therapy_wip", "en", "clinical/models")
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

val data = Seq("The had previously undergone a left mastectomy and an axillary lymph node dissection for a left breast cancer twenty years ago.
The tumor was positive for ER and PR. Postoperatively, radiotherapy was administered to her breast.
The cancer recurred as a right lung metastasis 13 years later. The patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses, as first line therapy.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
| chunk                          | ner_label             |
|:-------------------------------|:----------------------|
| mastectomy                     | Cancer_Surgery        |
| axillary lymph node dissection | Cancer_Surgery        |
| radiotherapy                   | Radiotherapy          |
| recurred                       | Response_To_Treatment |
| adriamycin                     | Chemotherapy          |
| 60 mg/m2                       | Dosage                |
| cyclophosphamide               | Chemotherapy          |
| 600 mg/m2                      | Dosage                |
| six courses                    | Cycle_Count           |
| first line                     | Line_Of_Therapy       |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_therapy_wip|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|869.8 KB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash

                label     tp    fp    fn  total  precision  recall   f1
         Cycle_Number   58.0  18.0  15.0   73.0       0.76    0.79 0.78
Response_To_Treatment  249.0  80.0 180.0  429.0       0.76    0.58 0.66
          Cycle_Count  151.0  48.0  24.0  175.0       0.76    0.86 0.81
   Unspecific_Therapy  167.0  88.0  67.0  234.0       0.65    0.71 0.68
         Chemotherapy  535.0  30.0  83.0  618.0       0.95    0.87 0.90
     Targeted_Therapy  144.0   9.0  35.0  179.0       0.94    0.80 0.87
         Radiotherapy  188.0  17.0  34.0  222.0       0.92    0.85 0.88
       Cancer_Surgery  526.0  60.0 119.0  645.0       0.90    0.82 0.85
      Line_Of_Therapy   73.0  14.0  14.0   87.0       0.84    0.84 0.84
     Hormonal_Therapy   95.0   1.0  21.0  116.0       0.99    0.82 0.90
        Immunotherapy   90.0  58.0  21.0  111.0       0.61    0.81 0.69
            Cycle_Day  149.0  33.0  34.0  183.0       0.82    0.81 0.82
            Frequency  287.0  35.0  62.0  349.0       0.89    0.82 0.86
                Route   82.0  17.0  15.0   97.0       0.83    0.85 0.84
             Duration  399.0  95.0 148.0  547.0       0.81    0.73 0.77
               Dosage  718.0  38.0 109.0  827.0       0.95    0.87 0.91
       Radiation_Dose   84.0  15.0  12.0   96.0       0.85    0.88 0.86
            macro_avg 3995.0 656.0 993.0 4988.0       0.84    0.81 0.82
            micro_avg    NaN   NaN   NaN    NaN       0.86    0.80 0.83
```