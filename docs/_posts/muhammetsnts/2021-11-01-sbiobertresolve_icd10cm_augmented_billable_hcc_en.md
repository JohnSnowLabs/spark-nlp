---
layout: model
title: Sentence Entity Resolver for Billable ICD10-CM HCC Codes
author: John Snow Labs
name: sbiobertresolve_icd10cm_augmented_billable_hcc
date: 2021-11-01
tags: [icd10cm, hcc, entity_resolution, licensed, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.1
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted medical entities to ICD10-CM codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings and it supports 7-digit codes with HCC status. It has been updated by dropping the invalid codes that exist in the previous versions. In the result, look for the `all_k_aux_labels` parameter in the metadata to get HCC status. The HCC status can be divided to get further information: `billable status`, `hcc status`, and `hcc score`. For reference: [please click here](http://www.formativhealth.com/wp-content/uploads/2018/06/HCC-White-Paper.pdf) .

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_augmented_billable_hcc_en_3.3.1_2.4_1635784379929.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_augmented_billable_hcc_en_3.3.1_2.4_1635784379929.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

```sbiobertresolve_icd10cm_augmented_billable_hcc``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_clinical``` as NER model. ```PROBLEM``` set in ```.setWhiteList()```.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
c2doc = Chunk2Doc()\
.setInputCols("ner_chunk")\
.setOutputCol("ner_chunk_doc") 

sbert_embedder = BertSentenceEmbeddings\
.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sentence_embeddings")

icd_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_icd10cm_augmented_billable_hcc", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sentence_embeddings"]) \
.setOutputCol("icd10cm_code")\
.setDistanceFunction("EUCLIDEAN")

resolver_pipeline = Pipeline(
stages = [
document_assembler,
sentenceDetectorDL,
tokenizer,
word_embeddings,
clinical_ner,
ner_converter_icd,
c2doc,
sbert_embedder,
icd_resolver
])

data_ner = spark.createDataFrame([["A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection."]]).toDF("text")

results =  resolver_pipeline.fit(data_ner).transform(data_ner)
```
```scala
...
val chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli","en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sbert_embeddings")

val icd10_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_icd10cm_augmented_billable_hcc","en", "clinical/models")
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
nlu.load("en.resolve.icd10cm.augmented_billable").predict("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection.""")
```

</div>

## Results

```bash
+-------------------------------------+-------+------------+----------------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------------------+
|                            ner_chunk| entity|icd10cm_code|                                                           resolutions|                                                             all_codes|                                                          billable_hcc|
+-------------------------------------+-------+------------+----------------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------------------+
|        gestational diabetes mellitus|PROBLEM|       O2441|gestational diabetes mellitus:::postpartum gestational diabetes mel...|    O2441:::O2443:::Z8632:::Z875:::O2431:::O2411:::O244:::O241:::O2481|0||0||0:::0||0||0:::1||0||0:::0||0||0:::0||0||0:::0||0||0:::0||0||0...|
|subsequent type two diabetes mellitus|PROBLEM|       O2411|pre-existing type 2 diabetes mellitus:::disorder associated with ty...|O2411:::E118:::E11:::E139:::E119:::E113:::E1144:::Z863:::Z8639:::E1...|0||0||0:::1||1||18:::0||0||0:::1||1||19:::1||1||19:::0||0||0:::1||1...|
|                                 T2DM|PROBLEM|         E11|t2dm [type 2 diabetes mellitus]:::tndm2:::t2 category:::sma2:::nf2:...|E11:::P702:::C801:::G121:::Q850:::C779:::C509:::C439:::E723:::C5700...|0||0||0:::1||0||0:::1||1||12:::1||1||72:::0||0||0:::1||1||10:::0||0...|
|             HTG-induced pancreatitis|PROBLEM|       K8520|alcohol-induced pancreatitis:::pancreatitis:::drug induced acute pa...|K8520:::K859:::K853:::K8590:::K85:::F102:::K858:::K8591:::K852:::K8...|1||0||0:::0||0||0:::0||0||0:::1||0||0:::0||0||0:::0||0||0:::0||0||0...|
|                      acute hepatitis|PROBLEM|        K720|acute hepatitis:::acute hepatitis a:::acute infectious hepatitis:::...|K720:::B15:::B179:::B172:::Z0389:::B159:::B150:::B16:::K752:::K712:...|0||0||0:::0||0||0:::1||0||0:::1||0||0:::1||0||0:::1||0||0:::1||0||0...|
|                              obesity|PROBLEM|        E669|obesity:::abdominal obesity:::obese:::central obesity:::overweight ...|E669:::E668:::Z6841:::Q130:::E66:::E6601:::Z8639:::E349:::H3550:::Z...|1||0||0:::1||0||0:::1||1||22:::1||0||0:::0||0||0:::1||1||22:::1||0|...|
|                    a body mass index|PROBLEM|       Z6841|finding of body mass index:::observation of body mass index:::mass ...|Z6841:::E669:::R229:::Z681:::R223:::R221:::Z68:::R222:::R220:::R418...|1||1||22:::1||0||0:::1||0||0:::1||0||0:::0||0||0:::1||0||0:::0||0||...|
|                             polyuria|PROBLEM|         R35|polyuria:::polyuric state:::polyuric state (disorder):::hematuria::...|R35:::R358:::E232:::R31:::R350:::R8299:::N401:::E723:::O048:::R300:...|0||0||0:::1||0||0:::1||1||23:::0||0||0:::1||0||0:::0||0||0:::1||0||...|
|                           polydipsia|PROBLEM|        R631|polydipsia:::psychogenic polydipsia:::primary polydipsia:::psychoge...|R631:::F6389:::E232:::F639:::O40:::G475:::M7989:::R632:::R061:::H53...|1||0||0:::1||1||nan:::1||1||23:::1||1||nan:::0||0||0:::0||0||0:::1|...|
|                        poor appetite|PROBLEM|        R630|poor appetite:::poor feeding:::bad taste in mouth:::unpleasant tast...|R630:::P929:::R438:::R432:::E86:::R196:::F520:::Z724:::R0689:::Z768...|1||0||0:::1||0||0:::1||0||0:::1||0||0:::0||0||0:::1||0||0:::1||0||0...|
|                             vomiting|PROBLEM|        R111|vomiting:::intermittent vomiting:::vomiting symptoms:::periodic vom...|       R111:::R11:::R1110:::G43A1:::P921:::P9209:::G43A:::R1113:::R110|0||0||0:::0||0||0:::1||0||0:::1||1||nan:::1||0||0:::1||0||0:::0||0|...|
|        a respiratory tract infection|PROBLEM|        J988|respiratory tract infection:::upper respiratory tract infection:::b...|J988:::J069:::A499:::J22:::J209:::Z593:::T17:::J0410:::Z1383:::J189...|1||0||0:::1||0||0:::1||0||0:::1||0||0:::1||0||0:::1||0||0:::0||0||0...|
+-------------------------------------+-------+------------+----------------------------------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_icd10cm_augmented_billable_hcc|
|Compatibility:|Healthcare NLP 3.3.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[icd10cm_code]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on 01 November 2021 ICD10CM Dataset.