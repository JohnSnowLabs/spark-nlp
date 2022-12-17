---
layout: model
title: Detect Entities Related to Cancer Therapies
author: John Snow Labs
name: ner_oncology_therapy
date: 2022-10-25
tags: [licensed, clinical, oncology, en, ner, treatment]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 4.0.0
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_therapy_en_4.0.0_3.0_1666718855759.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner = MedicalNerModel.pretrained("ner_oncology_therapy", "en", "clinical/models") \
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

data = spark.createDataFrame([["The had previously undergone a left mastectomy and an axillary lymph node dissection for a left breast cancer twenty years ago.
The tumor was positive for ER and PR. Postoperatively, radiotherapy was administered to her breast.
The cancer recurred as a right lung metastasis 13 years later. The patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses, as first line therapy."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    
val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
    .setInputCols("document")
    .setOutputCol("sentence")
    
val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")
    
val word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")                
    
val ner = MedicalNerModel.pretrained("ner_oncology_therapy", "en", "clinical/models")
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
|Model Name:|ner_oncology_therapy|
|Compatibility:|Spark NLP for Healthcare 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|34.4 MB|
|Dependencies:|embeddings_clinical|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
                label   tp   fp   fn  total  precision  recall   f1
         Cycle_Number   78   41   19     97       0.66    0.80 0.72
Response_To_Treatment  451  205  145    596       0.69    0.76 0.72
          Cycle_Count  210   75   20    230       0.74    0.91 0.82
   Unspecific_Therapy  189   76   89    278       0.71    0.68 0.70
         Chemotherapy  831   87   48    879       0.91    0.95 0.92
     Targeted_Therapy  194   28   34    228       0.87    0.85 0.86
         Radiotherapy  279   35   31    310       0.89    0.90 0.89
       Cancer_Surgery  720  192   99    819       0.79    0.88 0.83
      Line_Of_Therapy   95    6   11    106       0.94    0.90 0.92
     Hormonal_Therapy  170    6   15    185       0.97    0.92 0.94
        Immunotherapy   96   17   32    128       0.85    0.75 0.80
            Cycle_Day  205   38   43    248       0.84    0.83 0.84
            Frequency  363   33   64    427       0.92    0.85 0.88
                Route   93    6   20    113       0.94    0.82 0.88
             Duration  527  102  234    761       0.84    0.69 0.76
               Dosage  959   63  101   1060       0.94    0.90 0.92
       Radiation_Dose  106   12   20    126       0.90    0.84 0.87
            macro_avg 5566 1022 1025   6591       0.85    0.84 0.84
            micro_avg 5566 1022 1025   6591       0.85    0.84 0.84
```
