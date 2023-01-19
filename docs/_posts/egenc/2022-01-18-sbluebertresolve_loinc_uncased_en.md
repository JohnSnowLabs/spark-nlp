---
layout: model
title: Sentence Entity Resolver for LOINC (sbluebert_base_uncased_mli embeddings)
author: John Snow Labs
name: sbluebertresolve_loinc_uncased
date: 2022-01-18
tags: [loinc, licensed, clinical, entity_resolution, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted clinical NER entities to LOINC codes using `sbluebert_base_uncased_mli` Sentence Bert Embeddings. It trained on the augmented version of the uncased (lowercased) dataset which is used in previous LOINC resolver models.

## Predicted Entities

`LOINC Code`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_LOINC_AUGMENTED/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbluebertresolve_loinc_uncased_en_3.3.4_2.4_1642535076764.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbluebertresolve_loinc_uncased_en_3.3.4_2.4_1642535076764.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
.setInputCols("document")\
.setOutputCol("sentence")

tokenizer = Tokenizer() \
.setInputCols(["document"]) \
.setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical','en', 'clinical/models')\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

ner = MedicalNerModel.pretrained("ner_radiology", "en", "clinical/models") \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")

ner_converter = NerConverter() \
.setInputCols(["sentence", "token", "ner"]) \
.setOutputCol("ner_chunk")\
.setWhiteList(['Test'])

chunk2doc = Chunk2Doc() \
.setInputCols("ner_chunk") \
.setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings.pretrained("sbluebert_base_uncased_mli", "en", "clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")\
.setCaseSensitive(True)

resolver = SentenceEntityResolverModel.pretrained("sbluebertresolve_loinc_uncased", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"])\
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

pipeline_loinc = Pipeline(stages = [
documentAssembler, 
sentenceDetector, 
tokenizer,  
word_embeddings, 
ner, 
ner_converter, 
chunk2doc, 
sbert_embedder, 
resolver
])

test = """The patient is a 22-year-old female with a history of obesity. She has a BMI of 33.5 kg/m2, aspartate aminotransferase 64, and alanine aminotransferase 126. Her hgba1c is 8.2%."""

model = pipeline_loinc.fit(spark.createDataFrame([['']]).toDF("text"))

sparkDF = spark.createDataFrame([[test]]).toDF("text")

result = model.transform(sparkDF)
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
.setInputCols("document")
.setOutputCol("sentence")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical","en", "clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_radiology", "en", "clinical/models") 
.setInputCols(Array("sentence", "token", "embeddings")) 
.setOutputCol("ner")

val ner_converter = NerConverter() 
.setInputCols(Array("sentence", "token", "ner")) 
.setOutputCol("ner_chunk")
.setWhiteList(Array("Test"))

val chunk2doc = Chunk2Doc() 
.setInputCols("ner_chunk") 
.setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbluebert_base_uncased_mli", "en", "clinical/models")
.setInputCols("ner_chunk_doc")
.setOutputCol("sbert_embeddings")
.setCaseSensitive(True)

val resolver = SentenceEntityResolverModel.pretrained("sbluebertresolve_loinc_uncased", "en", "clinical/models") 
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val pipeline_loinc = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, ner, ner_converter, chunk2doc, sbert_embedder, resolver))

val data = Seq("The patient is a 22-year-old female with a history of obesity. She has a BMI of 33.5 kg/m2, aspartate aminotransferase 64, and alanine aminotransferase 126. Her hgba1c is 8.2%.").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.loinc_uncased").predict("""The patient is a 22-year-old female with a history of obesity. She has a BMI of 33.5 kg/m2, aspartate aminotransferase 64, and alanine aminotransferase 126. Her hgba1c is 8.2%.""")
```

</div>

## Results

```bash
+-------------------------------------+------+-----------+----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                            ner_chunk|entity| resolution|                                           all_codes|                                                                                                                                                                                             resolutions|
+-------------------------------------+------+-----------+----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                  BMI|  Test|    39156-5|[39156-5, LP35925-4, BDYCRC, 73964-9, 59574-4,...]  |[Body mass index, Body mass index (BMI), Body circumference, Body muscle mass, Body mass index (BMI) [Percentile], ...]                                                                                 |
|           aspartate aminotransferase|  Test|    14409-7|['14409-7', '16325-3', '1916-6', '16324-6',...]     |['Aspartate aminotransferase', 'Alanine aminotransferase/Aspartate aminotransferase', 'Aspartate aminotransferase/Alanine aminotransferase', 'Alanine aminotransferase', ...]                           |
|             alanine aminotransferase|  Test|    16324-6|['16324-6', '1916-6', '16325-3', '59245-1',...]     |['Alanine aminotransferase', 'Aspartate aminotransferase/Alanine aminotransferase', 'Alanine aminotransferase/Aspartate aminotransferase', 'Alanine glyoxylate aminotransferase',...]                   |
|                               hgba1c|  Test|    41995-2|['41995-2', 'LP35944-5', 'LP19717-5', '43150-2',...]|['Hemoglobin A1c', 'HbA1c measurement device', 'HBA1 gene', 'HbA1c measurement device panel', ...]                                                                                                      |
+-------------------------------------+------+-----------+------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbluebertresolve_loinc_uncased|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[loinc_code]|
|Language:|en|
|Size:|653.4 MB|
|Case sensitive:|false|
|Dependencies:|sbluebert_base_uncased_mli|

## Data Source

Trained on standard LOINC coding system.