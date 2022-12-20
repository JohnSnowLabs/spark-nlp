---
layout: model
title: Extract Oncology Tests
author: John Snow Labs
name: ner_oncology_test_wip
date: 2022-09-30
tags: [licensed, clinical, oncology, en, ner, test]
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

This model extracts mentions of tests from oncology texts, including pathology tests and imaging tests.

Definitions of Predicted Entities:

- `Biomarker`: Biological molecules that indicate the presence or absence of cancer, or the type of cancer. Oncogenes are excluded from this category.
- `Biomarker_Result`: Terms or values that are identified as the result of a biomarkers.
- `Imaging_Test`: Imaging tests mentioned in texts, such as "chest CT scan".
- `Oncogene`: Mentions of genes that are implicated in the etiology of cancer.
- `Pathology_Test`: Mentions of biopsies or tests that use tissue samples.


## Predicted Entities

`Biomarker`, `Biomarker_Result`, `Imaging_Test`, `Oncogene`, `Pathology_Test`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_test_wip_en_4.0.0_3.0_1664562782979.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner = MedicalNerModel.pretrained("ner_oncology_test_wip", "en", "clinical/models") \
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

data = spark.createDataFrame([["A biopsy was conducted using an ultrasound guided thick-needle. His chest computed tomography (CT) scan was negative."]]).toDF("text")

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
    
val ner = MedicalNerModel.pretrained("ner_oncology_test_wip", "en", "clinical/models")
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

val data = Seq("A biopsy was conducted using an ultrasound guided thick-needle. His chest computed tomography (CT) scan was negative.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)

```
</div>

## Results

```bash
| chunk                     | ner_label      |
|:--------------------------|:---------------|
| biopsy                    | Pathology_Test |
| ultrasound                | Imaging_Test   |
| chest computed tomography | Imaging_Test   |
| CT                        | Imaging_Test   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_test_wip|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|858.2 KB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
           label     tp    fp    fn  total  precision  recall   f1
    Imaging_Test 1518.0 156.0 191.0 1709.0       0.91    0.89 0.90
Biomarker_Result  861.0 145.0 245.0 1106.0       0.86    0.78 0.82
  Pathology_Test  600.0 105.0 209.0  809.0       0.85    0.74 0.79
       Biomarker  917.0 166.0 194.0 1111.0       0.85    0.83 0.84
        Oncogene  274.0  84.0  83.0  357.0       0.77    0.77 0.77
       macro_avg 4170.0 656.0 922.0 5092.0       0.85    0.80 0.82
       micro_avg    NaN   NaN   NaN    NaN       0.86    0.82 0.84
```