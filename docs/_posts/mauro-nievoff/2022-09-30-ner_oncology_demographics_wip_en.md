---
layout: model
title: Extract Demographic Entities from Oncology Texts
author: John Snow Labs
name: ner_oncology_demographics_wip
date: 2022-09-30
tags: [licensed, clinical, oncology, en, ner, demographics]
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

This model extracts demographic information from oncology texts, including age, gender and smoking status.

Definitions of Predicted Entities:

- `Age`: All mention of ages, past or present, related to the patient or with anybody else.
- `Gender`: Gender-specific nouns and pronouns (including words such as "him" or "she", and family members such as "father").
- `Race_Ethnicity`: The race and ethnicity categories include racial and national origin or sociocultural groups.
- `Smoking_Status`: All mentions of smoking related to the patient or to someone else.

## Predicted Entities

 `Age`, `Gender`, `Race_Ethnicity`, `Smoking_Status`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_demographics_wip_en_4.0.0_3.0_1664563557899.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_demographics_wip_en_4.0.0_3.0_1664563557899.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    .setOutputCol("token")\
    .setSplitChars(['-'])

word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")                

ner = MedicalNerModel.pretrained("ner_oncology_demographics_wip", "en", "clinical/models") \
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

data = spark.createDataFrame([["The patient is a 40-year-old man with history of heavy smoking."]]).toDF("text")

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
    .setSplitChars(["-"])
    
val word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")                
    
val ner = MedicalNerModel.pretrained("ner_oncology_demographics_wip", "en", "clinical/models")
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

val data = Seq("The patient is a 40-year-old man with history of heavy smoking.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
| chunk       | ner_label      |
|:------------|:---------------|
| 40-year-old | Age            |
| man         | Gender         |
| smoking     | Smoking_Status |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_demographics_wip|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|849.2 KB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
         label     tp   fp   fn  total  precision  recall   f1
Smoking_Status   43.0 12.0 11.0   54.0       0.78    0.80 0.79
           Age  679.0 27.0 17.0  696.0       0.96    0.98 0.97
Race_Ethnicity   44.0  7.0  7.0   51.0       0.86    0.86 0.86
        Gender  933.0 14.0  8.0  941.0       0.99    0.99 0.99
     macro_avg 1699.0 60.0 43.0 1742.0       0.90    0.91 0.90
     micro_avg    NaN  NaN  NaN    NaN       0.97    0.98 0.97
```