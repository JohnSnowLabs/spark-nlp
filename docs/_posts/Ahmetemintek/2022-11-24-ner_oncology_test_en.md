---
layout: model
title: Extract Oncology Tests
author: John Snow Labs
name: ner_oncology_test
date: 2022-11-24
tags: [licensed, clinical, oncology, en, ner, test]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.2.2
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_test_en_4.2.2_3.0_1669307746859.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_test_en_4.2.2_3.0_1669307746859.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner = MedicalNerModel.pretrained("ner_oncology_test", "en", "clinical/models") \
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
    .setInputCols("document")
    .setOutputCol("sentence")
    
val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")
    
val word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")                
    
val ner = MedicalNerModel.pretrained("ner_oncology_test", "en", "clinical/models")
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
| chunk                          | ner_label      |
|:-------------------------------|:---------------|
| biopsy                         | Pathology_Test |
| ultrasound guided thick-needle | Pathology_Test |
| chest computed tomography      | Imaging_Test   |
| CT                             | Imaging_Test   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_test|
|Compatibility:|Healthcare NLP 4.2.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|34.2 MB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
           label   tp   fp  fn  total  precision  recall   f1
    Imaging_Test 2020  229 184   2204       0.90    0.92 0.91
Biomarker_Result 1177  186 268   1445       0.86    0.81 0.84
  Pathology_Test  888  276 162   1050       0.76    0.85 0.80
       Biomarker 1287  254 228   1515       0.84    0.85 0.84
        Oncogene  365   89  84    449       0.80    0.81 0.81
       macro_avg 5737 1034 926   6663       0.83    0.85 0.84
       micro_avg 5737 1034 926   6663       0.85    0.86 0.85
```