---
layout: model
title: Sentence Entity Resolver for ICD10-CM (Augmented)
author: John Snow Labs
name: sbiobertresolve_icd10cm_augmented
date: 2022-01-21
tags: [icd10cm, entity_resolution, clinical, en, licensed]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.1
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted medical entities to ICD10-CM codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. Also, it has been augmented with synonyms for making it more accurate.

## Predicted Entities

`ICD10CM Codes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_augmented_en_3.3.1_3.0_1642756161477.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_augmented_en_3.3.1_3.0_1642756161477.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

icd10_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_icd10cm_augmented","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icd10_resolver])

data_ner = spark.createDataFrame([["A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection."]]).toDF("text")

results = nlpPipeline.fit(data_ner).transform(data_ner)
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


val clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
.setInputCols(Array("sentence","token","embeddings"))
.setOutputCol("ner")

val ner_converter = NerConverter()
.setInputCols(Array("sentence","token","ner"))
.setOutputCol("ner_chunk")
.setWhiteList(Array('PROBLEM'))


val chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli","en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sbert_embeddings")

val icd10_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_icd10cm_augmented","en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icd10_resolver))

val data = Seq("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection.").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.icd10cm.augmented").predict("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection.""")
```

</div>

## Results

```bash
+-------------------------------------+-------+------------+----------------------------------------------------------------------+----------------------------------------------------------------------+
|                            ner_chunk| entity|icd10cm_code|                                                           resolutions|                                                             all_codes|
+-------------------------------------+-------+------------+----------------------------------------------------------------------+----------------------------------------------------------------------+
|        gestational diabetes mellitus|PROBLEM|       O2441|gestational diabetes mellitus:::postpartum gestational diabetes mel...|    O2441:::O2443:::Z8632:::Z875:::O2431:::O2411:::O244:::O241:::O2481|
|subsequent type two diabetes mellitus|PROBLEM|       O2411|pre-existing type 2 diabetes mellitus:::disorder associated with ty...|O2411:::E118:::E11:::E139:::E119:::E113:::E1144:::Z863:::Z8639:::E1...|
|                                 T2DM|PROBLEM|         E11|type 2 diabetes mellitus:::disorder associated with type 2 diabetes...|E11:::E118:::E119:::O2411:::E109:::E139:::E113:::E8881:::Z833:::D64...|
|             HTG-induced pancreatitis|PROBLEM|       K8520|alcohol-induced pancreatitis:::drug-induced acute pancreatitis:::he...|K8520:::K853:::K8590:::F102:::K852:::K859:::K8580:::K8591:::K858:::...|
|                      acute hepatitis|PROBLEM|        K720|acute hepatitis:::acute hepatitis a:::acute infectious hepatitis:::...|K720:::B15:::B179:::B172:::Z0389:::B159:::B150:::B16:::K752:::K712:...|
|                              obesity|PROBLEM|        E669|obesity:::abdominal obesity:::obese:::central obesity:::overweight ...|E669:::E668:::Z6841:::Q130:::E66:::E6601:::Z8639:::E349:::H3550:::Z...|
|                    a body mass index|PROBLEM|       Z6841|finding of body mass index:::observation of body mass index:::mass ...|Z6841:::E669:::R229:::Z681:::R223:::R221:::Z68:::R222:::R220:::R418...|
|                             polyuria|PROBLEM|         R35|polyuria:::nocturnal polyuria:::polyuric state:::polyuric state (di...|R35:::R3581:::R358:::E232:::R31:::R350:::R8299:::N401:::E723:::O048...|
|                           polydipsia|PROBLEM|        R631|polydipsia:::psychogenic polydipsia:::primary polydipsia:::psychoge...|R631:::F6389:::E232:::F639:::O40:::G475:::M7989:::R632:::R061:::H53...|
|                        poor appetite|PROBLEM|        R630|poor appetite:::poor feeding:::bad taste in mouth:::unpleasant tast...|R630:::P929:::R438:::R432:::E86:::R196:::F520:::Z724:::R0689:::Z768...|
|                             vomiting|PROBLEM|        R111|vomiting:::intermittent vomiting:::vomiting symptoms:::periodic vom...|       R111:::R11:::R1110:::G43A1:::P921:::P9209:::G43A:::R1113:::R110|
|        a respiratory tract infection|PROBLEM|        J988|respiratory tract infection:::upper respiratory tract infection:::b...|J988:::J069:::A499:::J22:::J209:::Z593:::T17:::J0410:::Z1383:::J189...|
+-------------------------------------+-------+------------+----------------------------------------------------------------------+----------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_icd10cm_augmented|
|Compatibility:|Healthcare NLP 3.3.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[icd10cm_code]|
|Language:|en|
|Size:|1.4 GB|
|Case sensitive:|false|

## Data Source

Trained on ICD10CM 2022 Codes dataset: https://www.cdc.gov/nchs/icd/icd10cm.htm
