---
layout: model
title: Extract Entities Related to TNM Staging
author: John Snow Labs
name: ner_oncology_tnm_wip
date: 2022-09-30
tags: [licensed, clinical, oncology, en, ner]
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

This model extracts staging information and mentions related to tumors, lymph nodes and metastases. Tumor_Description is used to extract characteristics from tumors such as size, histological type or presence of invasion. Lymph_Node_Modifier is used to extract modifiers that refer to an abnormal lymph node (such as "enlarged").

## Predicted Entities

`Lymph_Node`, `Staging`, `Lymph_Node_Modifier`, `Tumor_Description`, `Tumor`, `Metastasis`, `Cancer_Dx`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_tnm_wip_en_4.0.0_3.0_1664561705395.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner = MedicalNerModel.pretrained("ner_oncology_tnm_wip", "en", "clinical/models") \
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

data = spark.createDataFrame([["The final diagnosis was metastatic breast carcinoma, and the TNM classification was T2N1M1 stage IV. The histological grade of this 4 cm tumor was grade 2."]]).toDF("text")

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
    
val ner = MedicalNerModel.pretrained("ner_oncology_tnm_wip", "en", "clinical/models")
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

val data = Seq("The final diagnosis was metastatic breast carcinoma, and the TNM classification was T2N1M1 stage IV. The histological grade of this 4 cm tumor was grade 2.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
| chunk              | ner_label         |
|:-------------------|:------------------|
| metastatic         | Metastasis        |
| breast carcinoma   | Cancer_Dx         |
| T2N1M1 stage IV    | Staging           |
| histological grade | Tumor_Description |
| 4 cm               | Tumor_Description |
| tumor              | Tumor             |
| grade 2            | Tumor_Description |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_tnm_wip|
|Compatibility:|Spark NLP for Healthcare 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|858.6 KB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
              label     tp    fp    fn  total  precision  recall   f1
         Lymph_Node  410.0  31.0 100.0  510.0       0.93    0.80 0.86
            Staging  166.0  15.0  50.0  216.0       0.92    0.77 0.84
Lymph_Node_Modifier   19.0   1.0  12.0   31.0       0.95    0.61 0.75
  Tumor_Description 1996.0 537.0 385.0 2381.0       0.79    0.84 0.81
              Tumor  834.0  48.0 108.0  942.0       0.95    0.89 0.91
         Metastasis  273.0  16.0  16.0  289.0       0.94    0.94 0.94
          Cancer_Dx  949.0  44.0 117.0 1066.0       0.96    0.89 0.92
          macro_avg 4647.0 692.0 788.0 5435.0       0.92    0.82 0.86
          micro_avg    NaN   NaN   NaN    NaN       0.88    0.86 0.86
```