---
layout: model
title: Sentence Entity Resolver for LOINC (sbluebert_base_uncased_mli embeddings)
author: John Snow Labs
name: sblubertresolve_loinc_uncased
date: 2021-12-24
tags: [en, licensed, entity_resolution, clinical, uncased, loinc]
task: Entity Resolution
language: en
edition: Spark NLP for Healthcare 3.3.4
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted clinical NER entities to LOINC codes using `sbluebert_base_uncased_mli` Sentence Bert Embeddings. It trained on the augmented version of the uncased (lowercased) dataset which is used in previous LOINC resolver models.

## Predicted Entities

`LOINC Code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sblubertresolve_loinc_uncased_en_3.3.4_2.4_1640342297007.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained()\
      .setInputCols("document")\
      .setOutputCol("sentence")

tokenizer = Tokenizer() \
      .setInputCols(["document"]) \
      .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical','en', 'clinical/models')\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

jsl_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
     .setInputCols(["sentence", "token", "embeddings"]) \
     .setOutputCol("jsl_ner")

jsl_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "jsl_ner"]) \
    .setOutputCol("jsl_ner_chunk")\
    .setWhiteList(['Test'])

chunk2doc = Chunk2Doc() \
    .setInputCols("jsl_ner_chunk") \
    .setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings.pretrained("sbluebert_base_uncased_mli", "en", "clinical/models")\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")\
     .setCaseSensitive(False)

resolver = SentenceEntityResolverModel.pretrained("sblubertresolve_loinc_uncased", "en", "clinical/models") \
      .setInputCols(["jsl_ner_chunk", "sbert_embeddings"])\
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")

pipeline_loinc = Pipeline(stages = [
    documentAssembler, 
    sentenceDetector, 
    tokenizer,  
    word_embeddings, 
    jsl_ner, 
    jsl_ner_converter, 
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
val documentAssembler = DocumentAssembler()\
          .setInputCol("text")\
          .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained()\
         .setInputCols("document")\
         .setOutputCol("sentence")

val tokenizer = Tokenizer() \
         .setInputCols("document") \
         .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical','en', 'clinical/models')\
         .setInputCols(Array("sentence", "token"))\
         .setOutputCol("embeddings")

val jsl_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
        .setInputCols(Array("sentence", "token", "embeddings")) \
        .setOutputCol("jsl_ner")

val jsl_ner_converter = NerConverter() \
        .setInputCols(Array("sentence", "token", "jsl_ner")) \
        .setOutputCol("jsl_ner_chunk")\
        .setWhiteList(Array('Test'))

val chunk2doc = Chunk2Doc() \
        .setInputCols("jsl_ner_chunk") \
        .setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbluebert_base_uncased_mli", "en", "clinical/models")\
         .setInputCols("ner_chunk_doc")\
         .setOutputCol("sbert_embeddings")\
         .setCaseSensitive(False)

val resolver = SentenceEntityResolverModel.pretrained("sblubertresolve_loinc_uncased", "en", "clinical/models") \
         .setInputCols(Array("jsl_ner_chunk", "sbert_embeddings"))\
         .setOutputCol("resolution")\
         .setDistanceFunction("EUCLIDEAN")

val pipeline_loinc = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, jsl_ner, jsl_ner_converter, chunk2doc, sbert_embedder, resolver))

val data = Seq("The patient is a 22-year-old female with a history of obesity. She has a BMI of 33.5 kg/m2, aspartate aminotransferase 64, and alanine aminotransferase 126. Her hgba1c is 8.2%.").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|    |   sent_id | ner_chunk                  | entity   | resolution   | all_codes                                                                                                                                                                                                                                                                                     | resolutions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|---:|----------:|:---------------------------|:---------|:-------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 |         1 | aspartate aminotransferase | Test     | 14409-7      | ['14409-7', '16325-3', '1916-6', '16324-6',...]                                              | ['Aspartate aminotransferase', 'Alanine aminotransferase/Aspartate aminotransferase', 'Aspartate aminotransferase/Alanine aminotransferase', 'Alanine aminotransferase', ...] |
|  1 |         1 | alanine aminotransferase   | Test     | 16324-6      | ['16324-6', '1916-6', '16325-3', '59245-1',...]                                           | ['Alanine aminotransferase', 'Aspartate aminotransferase/Alanine aminotransferase', 'Alanine aminotransferase/Aspartate aminotransferase', 'Alanine glyoxylate aminotransferase',...]         |
|  2 |         2 | hgba1c                     | Test     | 41995-2      | ['41995-2', 'LP35944-5', 'LP19717-5', '43150-2',...] | ['Hemoglobin A1c', 'HbA1c measurement device', 'HBA1 gene', 'HbA1c measurement device panel', ...]                                                                                                                                                                                                   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sblubertresolve_loinc_uncased|
|Compatibility:|Spark NLP for Healthcare 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[loinc_code]|
|Language:|en|
|Size:|647.9 MB|
|Case sensitive:|false|

## Data Source

Trained on standard LOINC coding system.