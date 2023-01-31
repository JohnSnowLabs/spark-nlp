---
layout: model
title: Sentence Entity Resolver for ICD-10-PCS (Augmented)
author: John Snow Labs
name: sbiobertresolve_icd10pcs_augmented
date: 2022-10-28
tags: [entity_resolution, clinical, en, licensed]
task: Entity Resolution
language: en
edition: Spark NLP for Healthcare 4.2.1
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted medical entities to ICD10-PCS codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It trained on the augmented version of the dataset which is used in previous ICD-10-PCS resolver model.

## Predicted Entities

`ICD-10-PCS Codes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10pcs_augmented_en_4.2.1_3.0_1666966980428.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10pcs_augmented_en_4.2.1_3.0_1666966980428.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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


clinical_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models")\
  .setInputCols(["sentence","token","embeddings"])\
  .setOutputCol("ner")

ner_converter = NerConverter()\
  .setInputCols(["sentence","token","ner"])\
  .setOutputCol("ner_chunk")\
  .setWhiteList(['Procedure', 'Test', 'Test_Result', 'Treatment', 'Pulse', 'Imaging_Technique', 'Labour_Delivery', 'Blood_Pressure', 'Oxygen_Therapy', 'Weight', 'LDL', 'O2_Saturation', 'BMI', 'Vaccine', 'Respiration', 'Temperature', 'Birth_Entity', 'Triglycerides'])


chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
  .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
  .setInputCols(["ner_chunk_doc"])\
  .setOutputCol("sbert_embeddings")

icd10pcs_resolver = SentenceEntityResolverModel\
  .pretrained("sbiobertresolve_icd10pcs_augmented","en", "clinical/models") \
  .setInputCols(["ner_chunk", "sbert_embeddings"]) \
  .setOutputCol("resolution")\
  .setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icd10pcs_resolver])


text = [["""Given the severity of her abdominal examination and her persistence of her symptoms, it is detected that need for laparoscopic appendectomy and possible open appendectomy as well as pyeloplasty. We recommend performing a mediastinoscopy"""]]


data= spark.createDataFrame(text).toDF('text')
results = nlpPipeline.fit(data).transform(data)

```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")


val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")


val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentence","token"))
.setOutputCol("embeddings")


val clinical_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models")
.setInputCols(Array("sentence","token","embeddings"))
.setOutputCol("ner")

val ner_converter = NerConverter()
.setInputCols(Array("sentence","token","ner"))
.setOutputCol("ner_chunk")
.setWhiteList(Array("Procedure", "Test", "Test_Result", "Treatment", "Pulse", "Imaging_Technique", "Labour_Delivery", "Blood_Pressure", "Oxygen_Therapy", "Weight", "LDL", "O2_Saturation", "BMI", "Vaccine", "Respiration", "Temperature", "Birth_Entity", "Triglycerides"))


val chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli","en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sbert_embeddings")

val icd10pcs_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_icd10pcs_augmented","en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icd10pcs_resolver))

val data = Seq("Given the severity of her abdominal examination and her persistence of her symptoms, it is detected that need for laparoscopic appendectomy and possible open appendectomy as well as pyeloplasty. We recommend performing a mediastinoscopy").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------------------+---------+-------------+------------------------------------------------------------+------------------------------------------------------------+
|                ner_chunk|   entity|icd10pcs_code|                                                 resolutions|                                                   all_codes|
+-------------------------+---------+-------------+------------------------------------------------------------+------------------------------------------------------------+
|    abdominal examination|     Test|      2W63XZZ|[traction of abdominal wall [traction of abdominal wall],...|[2W63XZZ, BW40ZZZ, DWY37ZZ, 0WJFXZZ, 2W03X2Z, 0WJF4ZZ, 0W...|
|laparoscopic appendectomy|Procedure|      0DTJ8ZZ|[resection of appendix, endo [resection of appendix, endo...|[0DTJ8ZZ, 0DT84ZZ, 0DTJ4ZZ, 0WBH4ZZ, 0DTR4ZZ, 0DBJ8ZZ, 0D...|
|        open appendectomy|Procedure|      0DBJ0ZZ|[excision of appendix, open approach [excision of appendi...|[0DBJ0ZZ, 0DTJ0ZZ, 0DBA0ZZ, 0D5J0ZZ, 0DB80ZZ, 0DB90ZZ, 04...|
|              pyeloplasty|Procedure|      0TS84ZZ|[reposition bilateral ureters, perc endo approach [reposi...|[0TS84ZZ, 0TS74ZZ, 069B3ZZ, 06SB3ZZ, 0TR74JZ, 0TQ43ZZ, 04...|
|          mediastinoscopy|Procedure|      BB1CZZZ|[fluoroscopy of mediastinum [fluoroscopy of mediastinum],...|[BB1CZZZ, 0WJC4ZZ, BB4CZZZ, 0WJC3ZZ, 0WHC33Z, 0WHC43Z, 0W...|
+-------------------------+---------+-------------+------------------------------------------------------------+------------------------------------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_icd10pcs_augmented|
|Compatibility:|Spark NLP for Healthcare 4.2.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[icd10pcs_code]|
|Language:|en|
|Size:|649.1 MB|
|Case sensitive:|false|

## References

Trained on ICD-10 Procedure Coding System dataset with sbiobert_base_cased_mli sentence embeddings. https://www.icd10data.com/ICD10PCS/Codes
