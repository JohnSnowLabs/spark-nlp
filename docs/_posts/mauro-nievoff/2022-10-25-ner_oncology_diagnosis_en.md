---
layout: model
title: Detect Entities Related to Cancer Diagnosis
author: John Snow Labs
name: ner_oncology_diagnosis
date: 2022-10-25
tags: [licensed, clinical, oncology, en, ner]
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

This model extracts entities related to cancer diagnosis, such as Metastasis, Histological_Type or Invasion.

Definitions of Predicted Entities:

- `Adenopathy`: Mentions of pathological findings of the lymph nodes.
- `Cancer_Dx`: Mentions of cancer diagnoses (such as "breast cancer") or pathological types that are usually used as synonyms for "cancer" (e.g. "carcinoma"). When anatomical references are present, they are included in the Cancer_Dx extraction.
- `Cancer_Score`: Clinical or imaging scores that are specific for cancer settings (e.g. "BI-RADS" or "Allred score").
- `Grade`: All pathological grading of tumors (e.g. "grade 1") or degrees of cellular differentiation (e.g. "well-differentiated")
- `Histological_Type`: Histological variants or cancer subtypes, such as "papillary", "clear cell" or "medullary". 
- `Invasion`: Mentions that refer to tumor invasion, such as "invasion" or "involvement". Metastases or lymph node involvement are excluded from this category.
- `Metastasis`: Terms that indicate a metastatic disease. Anatomical references are not included in these extractions.
- `Pathology_Result`: The findings of a biopsy from the pathology report that is not covered by another entity (e.g. "malignant ductal cells").
- `Performance_Status`: Mentions of performance status scores, such as ECOG and Karnofsky. The name of the score is extracted together with the result (e.g. "ECOG performance status of 4").
- `Staging`: Mentions of cancer stage such as "stage 2b" or "T2N1M0". It also includes words such as "in situ", "early-stage" or "advanced".
- `Tumor_Finding`: All nonspecific terms that may be related to tumors, either malignant or benign (for example: "mass", "tumor", "lesion", or "neoplasm").
- `Tumor_Size`: Size of the tumor, including numerical value and unit of measurement (e.g. "3 cm").



## Predicted Entities

`Adenopathy`, `Cancer_Dx`, `Cancer_Score`, `Grade`, `Histological_Type`, `Invasion`, `Metastasis`, `Pathology_Result`, `Performance_Status`, `Staging`, `Tumor_Finding`, `Tumor_Size` 


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_diagnosis_en_4.0.0_3.0_1666719602276.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_diagnosis_en_4.0.0_3.0_1666719602276.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner = MedicalNerModel.pretrained("ner_oncology_diagnosis", "en", "clinical/models") \
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

data = spark.createDataFrame([["Two years ago, the patient presented with a tumor in her left breast and adenopathies. She was diagnosed with invasive ductal carcinoma.
Last week she was also found to have a lung metastasis."]]).toDF("text")

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
    
val ner = MedicalNerModel.pretrained("ner_oncology_diagnosis", "en", "clinical/models")
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

val data = Seq("Two years ago, the patient presented with a tumor in her left breast and adenopathies. She was diagnosed with invasive ductal carcinoma.
Last week she was also found to have a lung metastasis.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
| chunk        | ner_label         |
|:-------------|:------------------|
| tumor        | Tumor_Finding     |
| adenopathies | Adenopathy        |
| invasive     | Histological_Type |
| ductal       | Histological_Type |
| carcinoma    | Cancer_Dx         |
| metastasis   | Metastasis        |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_diagnosis|
|Compatibility:|Spark NLP for Healthcare 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|34.3 MB|
|Dependencies:|embeddings_clinical|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
             label   tp   fp  fn  total  precision  recall   f1
 Histological_Type  354   63  99    453       0.85    0.78 0.81
           Staging  234   27  24    258       0.90    0.91 0.90
      Cancer_Score   36   15  26     62       0.71    0.58 0.64
     Tumor_Finding 1121   83 136   1257       0.93    0.89 0.91
          Invasion  154   27  27    181       0.85    0.85 0.85
        Tumor_Size 1058  126  71   1129       0.89    0.94 0.91
        Adenopathy   66   10  30     96       0.87    0.69 0.77
Performance_Status  116   15  19    135       0.89    0.86 0.87
  Pathology_Result  852  686 290   1142       0.55    0.75 0.64
        Metastasis  356   15  14    370       0.96    0.96 0.96
         Cancer_Dx 1302   88  92   1394       0.94    0.93 0.94
             Grade  201   23  35    236       0.90    0.85 0.87
         macro_avg 5850 1178 863   6713       0.85    0.83 0.84
         micro_avg 5850 1178 863   6713       0.85    0.87 0.86
```
