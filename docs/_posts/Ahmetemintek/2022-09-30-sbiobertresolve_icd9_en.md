---
layout: model
title: Sentence Entity Resolver for ICD-9-CM
author: John Snow Labs
name: sbiobertresolve_icd9
date: 2022-09-30
tags: [entity_resolution, en, licensed, icd9, clinical]
task: Entity Resolution
language: en
edition: Healthcare NLP 4.1.0
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted medical entities to ICD-9-CM codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings.

## Predicted Entities

`ICD-9-CM Codes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd9_en_4.1.0_3.0_1664533186655.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenizer = Tokenizer()\
  .setInputCols(["sentence"])\
  .setOutputCol("token")


word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence","token"])\
  .setOutputCol("embeddings")


clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")\
  .setInputCols(["sentence","token","embeddings"])\
  .setOutputCol("ner")

ner_converter = NerConverter()\
  .setInputCols(["sentence","token","ner"])\
  .setOutputCol("ner_chunk")\
  .setWhiteList(['PROBLEM'])

chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
  .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
  .setInputCols(["ner_chunk_doc"])\
  .setOutputCol("sbert_embeddings")

icd9_resolver = SentenceEntityResolverModel\
  .pretrained("sbiobertresolve_icd9","en", "clinical/models") \
  .setInputCols(["ner_chunk", "sbert_embeddings"]) \
  .setOutputCol("resolution")\
  .setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icd9_resolver])


clinical_note = [
    'A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years '
    'prior to presentation and subsequent type two diabetes mellitus, associated '
    'with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, ']


data= spark.createDataFrame([clinical_note]).toDF('text')
results = nlpPipeline.fit(data).transform(data)

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


val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence","token"))
  .setOutputCol("embeddings")


val clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence","token","embeddings"))
  .setOutputCol("ner")

val ner_converter = new NerConverter()
  .setInputCols(Array("sentence","token","ner"))
  .setOutputCol("ner_chunk")
  .setWhiteList(Array("PROBLEM"))

val chunk2doc = new Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
  .pretrained("sbiobert_base_cased_mli","en","clinical/models")
  .setInputCols(Array("ner_chunk_doc"))
  .setOutputCol("sbert_embeddings")

val icd9_resolver = SentenceEntityResolverModel
  .pretrained("sbiobertresolve_icd9","en", "clinical/models") 
  .setInputCols(Array("ner_chunk", "sbert_embeddings")) 
  .setOutputCol("resolution")
  .setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icd9_resolver))

val data = Seq("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus, associated with an acute hepatitis and obesity with a body mass index (BMI) of 33.5 kg/m2").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)

```
</div>

## Results

```bash
+-------------------------------------+-------+---------+--------------------------------------------------------------------------------+--------------------------------------------------------------------------------+
|                            ner_chunk| entity|icd9_code|                                                                      resolution|                                                                       all_codes|
+-------------------------------------+-------+---------+--------------------------------------------------------------------------------+--------------------------------------------------------------------------------+
|        gestational diabetes mellitus|PROBLEM|   V12.21|[Personal history of gestational diabetes, Neonatal diabetes mellitus, Second...|[V12.21, 775.1, 249, 250, 249.7, 249.71, 249.9, 249.61, 648.0, 249.51, 249.11...|
|subsequent type two diabetes mellitus|PROBLEM|      249|[Secondary diabetes mellitus, Diabetes mellitus, Secondary diabetes mellitus ...|[249, 250, 249.9, 249.7, 775.1, 249.6, 249.8, V12.21, 249.71, V77.1, 249.5, 2...|
|                   an acute hepatitis|PROBLEM|    571.1|[Acute alcoholic hepatitis, Viral hepatitis, Autoimmune hepatitis, Injury to ...|[571.1, 070, 571.42, 902.22, 279.51, 571.4, 091.62, 572.2, 864, 070.0, 572.0,...|
|                              obesity|PROBLEM|    278.0|[Overweight and obesity, Morbid obesity, Overweight, Screening for obesity, O...|[278.0, 278.01, 278.02, V77.8, 278, 278.00, 272.2, 783.1, 277.7, 728.5, 521.5...|
|                    a body mass index|PROBLEM|      V85|[Body mass index [BMI], Human bite, Localized adiposity, Effects of air press...|[V85, E928.3, 278.1, 993, E008.4, V61.5, 747.63, V85.5, 278.02, 780.97, 782.8...|
+-------------------------------------+-------+---------+--------------------------------------------------------------------------------+--------------------------------------------------------------------------------+
only showing top 5 rows
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_icd9|
|Compatibility:|Healthcare NLP 4.1.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[icd9_code]|
|Language:|en|
|Size:|50.1 MB|
|Case sensitive:|false|
