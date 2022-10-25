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
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts entities related to oncology therapies using granular labels, including mentions of treatments, posology information and line of therapy.

## Predicted Entities

`Cycle_Number`, `Response_To_Treatment`, `Cycle_Count`, `Unspecific_Therapy`, `Chemotherapy`, `Targeted_Therapy`, `Radiotherapy`, `Cancer_Surgery`, `Line_Of_Therapy`, `Hormonal_Therapy`, `Immunotherapy`, `Cycle_Day`, `Frequency`, `Route`, `Duration`, `Dosage`, `Radiation_Dose`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
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
                label     tp     fp     fn  total  precision  recall   f1
         Cycle_Number   78.0   41.0   19.0   97.0       0.66    0.80 0.72
Response_To_Treatment  451.0  205.0  145.0  596.0       0.69    0.76 0.72
          Cycle_Count  210.0   75.0   20.0  230.0       0.74    0.91 0.82
   Unspecific_Therapy  189.0   76.0   89.0  278.0       0.71    0.68 0.70
         Chemotherapy  831.0   87.0   48.0  879.0       0.91    0.95 0.92
     Targeted_Therapy  194.0   28.0   34.0  228.0       0.87    0.85 0.86
         Radiotherapy  279.0   35.0   31.0  310.0       0.89    0.90 0.89
       Cancer_Surgery  720.0  192.0   99.0  819.0       0.79    0.88 0.83
      Line_Of_Therapy   95.0    6.0   11.0  106.0       0.94    0.90 0.92
     Hormonal_Therapy  170.0    6.0   15.0  185.0       0.97    0.92 0.94
        Immunotherapy   96.0   17.0   32.0  128.0       0.85    0.75 0.80
            Cycle_Day  205.0   38.0   43.0  248.0       0.84    0.83 0.84
            Frequency  363.0   33.0   64.0  427.0       0.92    0.85 0.88
                Route   93.0    6.0   20.0  113.0       0.94    0.82 0.88
             Duration  527.0  102.0  234.0  761.0       0.84    0.69 0.76
               Dosage  959.0   63.0  101.0 1060.0       0.94    0.90 0.92
       Radiation_Dose  106.0   12.0   20.0  126.0       0.90    0.84 0.87
            macro_avg 5566.0 1022.0 1025.0 6591.0       0.85    0.84 0.84
            micro_avg    NaN    NaN    NaN    NaN       0.85    0.84 0.84
```