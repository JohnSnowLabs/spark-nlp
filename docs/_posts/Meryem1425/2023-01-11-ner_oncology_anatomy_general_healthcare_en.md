---
layout: model
title: Extract Anatomical Entities from Oncology Texts
author: John Snow Labs
name: ner_oncology_anatomy_general_healthcare
date: 2023-01-11
tags: [licensed, clinical, oncology, en, ner, anatomy]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts anatomical entities using an unspecific label.

## Predicted Entities

`Anatomical_Site`, `Direction`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_anatomy_general_healthcare_en_4.2.4_3.0_1673477824696.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_anatomy_general_healthcare_en_4.2.4_3.0_1673477824696.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel\
    .pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel()\
    .pretrained("embeddings_healthcare_100d", "en", "clinical/models")\
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")   

ner = MedicalNerModel\
    .pretrained("ner_oncology_anatomy_general_healthcare", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverterInternal() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")
        
pipeline = Pipeline(stages=[document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter])

data = spark.createDataFrame([["The patient presented a mass in her left breast, and a possible metastasis in her lungs and in her liver."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
    
val sentence_detector = SentenceDetectorDLModel
      .pretrained("sentence_detector_dl_healthcare","en","clinical/models")
      .setInputCols("document")
      .setOutputCol("sentence")
    
val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")
    
val word_embeddings = WordEmbeddingsModel()
      .pretrained("embeddings_healthcare_100d", "en", "clinical/models")
      .setInputCols(Array("sentence", "token"))
      .setOutputCol("embeddings")                
    
val ner = MedicalNerModel.pretrained("ner_oncology_anatomy_general_healthcare", "en", "clinical/models")
      .setInputCols(Array("sentence", "token", "embeddings"))
      .setOutputCol("ner")

val ner_converter = new NerConverterInternal()
      .setInputCols(Array("sentence", "token", "ner"))
      .setOutputCol("ner_chunk")
        
val pipeline = new Pipeline().setStages(Array(document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter))    

val data = Seq("The patient presented a mass in her left breast, and a possible metastasis in her lungs and in her liver.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
| chunk   | ner_label       |
|:--------|:----------------|
| left    | Direction       |
| breast  | Anatomical_Site |
| lungs   | Anatomical_Site |
| liver   | Anatomical_Site |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_anatomy_general_healthcare|
|Compatibility:|Healthcare NLP 4.2.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|34.0 MB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
          label   tp  fp  fn  total  precision  recall   f1
Anatomical_Site 1439 235 333   1772       0.86    0.81 0.84
      Direction  434  92  65    499       0.83    0.87 0.85
      macro-avg 1873 327 398   2271       0.84    0.84 0.84
      micro-avg 1873 327 398   2271       0.85    0.82 0.84
```