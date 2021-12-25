---
layout: model
title: Sentence Entity Resolver for LOINC (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_loinc_cased
date: 2021-12-24
tags: [en, clinical, licensed, entity_resolution, loinc]
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

This model maps extracted clinical NER entities to LOINC codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It is trained with augmented cased (unlowered) concept names since sbiobert model is cased.

## Predicted Entities

`LOINC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_loinc_cased_en_3.3.4_2.4_1640374998947.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")

tokenizer = Tokenizer() \
      .setInputCols(["sentence"]) \
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

sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
    .setInputCols(["ner_chunk_doc"])\
    .setOutputCol("sbert_embeddings")\
    .setCaseSensitive(True)

resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_loinc_cased", "en", "clinical/models") \
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

test = 'The patient is a 22-year-old female with a history of obesity. She has a Body Mass Index of 33.5 kg/m2, Aspartate Aminotransferase 64, and Alanine Aminotransferase 126. Her Hemoglabin is 8.2%.'
model = pipeline_loinc.fit(spark.createDataFrame([['']]).toDF("text"))

sparkDF = spark.createDataFrame([[test]]).toDF("text")
result = model.transform(sparkDF)
```
```scala
val documentAssembler = DocumentAssembler()\
          .setInputCol("text")\
          .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained()\
          .setInputCols(Array("document"))\
          .setOutputCol("sentence")

val tokenizer = Tokenizer() \
          .setInputCols(Array("document")) \
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
         .setWhiteList(Array(Test'))

val chunk2doc = Chunk2Doc() \
        .setInputCols("jsl_ner_chunk") \
        .setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
        .setInputCols(Array("ner_chunk_doc"))\
        .setOutputCol("sbert_embeddings")\
        .setCaseSensitive(True)

val resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_loinc_cased", "en", "clinical/models") \
        .setInputCols(Array("jsl_ner_chunk", "sbert_embeddings"))\
        .setOutputCol("resolution")\
        .setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, jsl_ner, jsl_ner_converter, chunk2doc,  sbert_embedder, resolver))

val data = Seq("The patient is a 22-year-old female with a history of obesity. She has a Body Mass Index of 33.5 kg/m2, Aspartate Aminotransferase 64, and Alanine Aminotransferase 126. Her Hemoglabin is 8.2%.").toDF("text")

val result = pipeline.fit(data).transform(data)


```
</div>

## Results

```bash
+-------------------------------------+------+-----------+----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                            ner_chunk|entity| resolution|                                           all_codes|                                                                                                                                                                                             resolutions|
+-------------------------------------+------+-----------+----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                      Body Mass Index|  Test|    39156-5|[39156-5, LP35925-4, BDYCRC, 73964-9, 59574-4,...   |[Body mass index, Body mass index (BMI), Body circumference, Body muscle mass, Body mass index (BMI) [Percentile], ...                                                                                  |
|           Aspartate Aminotransferase|  Test|    14409-7|[14409-7, 1916-6, 16325-3, 16324-6, 43822-6, 308... |[Aspartate aminotransferase, Aspartate aminotransferase/Alanine aminotransferase, Alanine aminotransferase/Aspartate aminotransferase, Alanine aminotransferase, Aspartate aminotransferase [Prese...   |
|             Alanine Aminotransferase|  Test|    16324-6|[16324-6, 16325-3, 14409-7, 1916-6, 59245-1, 30...  |[Alanine aminotransferase, Alanine aminotransferase/Aspartate aminotransferase, Aspartate aminotransferase, Aspartate aminotransferase/Alanine aminotransferase, Alanine glyoxylate aminotransfer,...   |
|                           Hemoglobin|  Test|    14775-1|[14775-1, 16931-8, 12710-0, 29220-1, 15082-1, 72... |[Hemoglobin, Hematocrit/Hemoglobin, Hemoglobin pattern, Haptoglobin, Methemoglobin, Oxyhemoglobin, Hemoglobin test status, Verdohemoglobin, Hemoglobin A, Hemoglobin distribution width, Myoglobin,...  |
+-------------------------------------+------+-----------+------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_loinc_cased|
|Compatibility:|Spark NLP for Healthcare 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[loinc_code]|
|Language:|en|
|Size:|648.5 MB|
|Case sensitive:|true|