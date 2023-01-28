---
layout: model
title: Sentence Entity Resolver for ICD10-CM (general 3 character codes)
author: John Snow Labs
name: sbiobertresolve_icd10cm_generalised
date: 2021-09-29
tags: [licensed, clinical, en, entity_resolution]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.2.1
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted medical entities to ICD10-CM codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It predicts ICD codes up to 3 characters (according to ICD10 code structure the first three characters represent general type of the injury or disease).

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_generalised_en_3.2.1_3.0_1632938859569.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_generalised_en_3.2.1_3.0_1632938859569.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

```sbiobertresolve_icd10cm_generalised``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_clinical``` as NER model. ```PROBLEM``` set in ```.setWhiteList()```.


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")

icd10_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_icd10cm_generalised","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icd10_resolver])

data = spark.createDataFrame([["This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU ."]]).toDF("text")

results = nlpPipeline.fit(data).transform(data)
```
```scala
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli","en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sbert_embeddings")

val icd10_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_icd10cm_generalised","en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icd10_resolver))

val data = Seq("This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.icd10cm_generalised").predict("""This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .""")
```

</div>

## Results

```bash
|    | chunk                       |   begin |   end | entity   | code   | code_desc                                                |   distance | all_k_resolutions                                                                                                                                                                                                                                                                                                                                       | all_k_codes                                                                 |
|---:|:----------------------------|--------:|------:|:---------|:-------|:---------------------------------------------------------|-----------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------|
|  0 | hypertension                |      68 |    79 | PROBLEM  | I10    | hypertension                                             |     0      | hypertension:::hypertension (high blood pressure):::h/o: hypertension:::fh: hypertension:::hypertensive heart disease:::labile hypertension:::history of hypertension (situation):::endocrine hypertension                                                                                                                                              | I10:::I15:::Z86:::Z82:::I11:::R03:::Z87:::E27                               |
|  1 | chronic renal insufficiency |      83 |   109 | PROBLEM  | N18    | chronic renal impairment                                 |     0.014  | chronic renal impairment:::renal insufficiency:::renal failure:::anaemia of chronic renal insufficiency:::impaired renal function disorder:::history of renal insufficiency:::prerenal renal failure:::abnormal renal function:::abnormal renal function                                                                                                | N18:::P96:::N19:::D63:::N28:::Z87:::N17:::N25:::R94                         |
|  2 | COPD                        |     113 |   116 | PROBLEM  | J44    | chronic obstructive lung disease (disorder)              |     0.1197 | chronic obstructive lung disease (disorder):::chronic obstructive pulmonary disease leaflet given:::chronic pulmonary congestion (disorder):::chronic respiratory failure (disorder):::chronic respiratory insufficiency:::cor pulmonale (chronic):::history of - chronic lung disease (situation)                                                      | J44:::Z76:::J81:::J96:::R06:::I27:::Z87                                     |
|  3 | gastritis                   |     120 |   128 | PROBLEM  | K29    | gastritis                                                |     0      | gastritis:::bacterial gastritis:::parasitic gastritis                                                                                                                                                                                                                                                                                                   | K29:::B96:::K93                                                             |
|  4 | TIA                         |     136 |   138 | PROBLEM  | S06    | cerebral concussion                                      |     0.1662 | cerebral concussion:::transient ischemic attack (disorder):::thalamic stroke:::cerebral trauma:::stroke:::traumatic amputation:::spinal cord stroke                                                                                                                                                                                                     | S06:::G45:::I63:::S09:::I64:::T14:::G95                                     |
|  5 | a non-ST elevation MI       |     182 |   202 | PROBLEM  | I21    | non-st elevation (nstemi) myocardial infarction          |     0.1615 | non-st elevation (nstemi) myocardial infarction:::nonruptured cerebral artery dissection:::acute stroke, nonatherosclerotic:::nontraumatic ischemic infarction of muscle, unsp shoulder:::history of nonatherosclerotic stroke without residual deficits:::non-traumatic cerebral hemorrhage                                                            | I21:::I67:::I63:::M62:::Z86:::I61                                           |
|  6 | Guaiac positive stools      |     208 |   229 | PROBLEM  | R85    | abnormal anal pap                                        |     0.1807 | abnormal anal pap:::straining at stool (finding):::amine test positive:::appendiceal colic:::fecal smearing:::epiploic appendagitis:::diverticulosis of intestine (finding):::appendicitis (disorder):::colostomy present (finding):::thickened anal verge (finding):::anal fissure:::amoebic enteritis:::zenkers diverticulum                          | R85:::R19:::Z78:::K38:::R15:::K65:::K57:::K37:::Z93:::K62:::K60:::A06:::K22 |
|  7 | mid LAD lesion              |     332 |   345 | PROBLEM  | I21    | stemi involving left anterior descending coronary artery |     0.1595 | stemi involving left anterior descending coronary artery:::divided left atrium:::disorder of left atrium:::double inlet left ventricle:::left os acromiale:::furuncle of left upper limb:::left anterior fascicular hemiblock (heart rhythm):::aberrant origin of left subclavian artery:::stent in circumflex branch of left coronary artery (finding) | I21:::Q24:::I51:::Q20:::M89:::L02:::I44:::Q27:::Z95                         |
|  8 | hypotension                 |     362 |   372 | PROBLEM  | I95    | hypotension                                              |     0      | hypotension:::supine hypotensive syndrome                                                                                                                                                                                                                                                                                                               | I95:::O26                                                                   |
|  9 | bradycardia                 |     378 |   388 | PROBLEM  | R00    | bradycardia                                              |     0      | bradycardia:::bradycardia (finding):::drug-induced bradycardia:::bradycardia (disorder)                                                                                                                                                                                                                                                                 | R00:::P29:::T50:::P20                                                       |
| 10 | vagal reaction              |     466 |   479 | PROBLEM  | G52    | vagus nerve finding                                      |     0.0926 | vagus nerve finding:::vasomotor reaction:::vesicular breathing (finding):::abdominal muscle tone - finding:::agonizing state:::paresthesia (finding):::glossolalia (finding):::tactile alteration (finding)                                                                                                                                             | G52:::I73:::R09:::R19:::R45:::R20:::R41:::R44                               |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_icd10cm_generalised|
|Compatibility:|Healthcare NLP 3.2.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_chunk_embeddings]|
|Output Labels:|[icd10cm_code]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on ICD10 Clinical Modification dataset with `sbiobert_base_cased_mli` sentence embeddings. https://www.icd10data.com/ICD10CM/Codes/
