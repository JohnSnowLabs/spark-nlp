---
layout: model
title:  ICD10CM Puerile Entity Resolver
author: John Snow Labs
name: chunkresolve_icd10cm_puerile_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-04-28
task: Entity Resolution
edition: Healthcare NLP 2.4.5
spark_version: 2.4
tags: [clinical,licensed,entity_resolution,en]
deprecated: true
annotator: ChunkEntityResolverModel
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance


## Predicted Entities
ICD10-CM Codes and their normalized definition with `clinical_embeddings`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_ICD10_CM.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_puerile_clinical_en_2.4.5_2.4_1588103916781.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
puerile_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_puerile_clinical","en","clinical/models")\
	.setInputCols("token","chunk_embeddings")\
	.setOutputCol("resolution")
pipeline_puerile = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, puerile_resolver])

model = pipeline_puerile.fit(spark.createDataFrame([["""The patient is a 5-month-old infant who presented initially on Monday with a cold, cough, and runny nose for 2 days. Mom states she had no fever. Her appetite was good but she was spitting up a lot. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed a right TM, which was red. Left TM was okay. She was fairly congested but looked happy and playful. She was started on Amoxil and Aldex and we told to recheck in 2 weeks to recheck her ear. Mom returned to clinic again today because she got much worse overnight. She was having difficulty breathing. She was much more congested and her appetite had decreased significantly today. She also spiked a temperature yesterday of 102.6 and always having trouble sleeping secondary to congestion."""]]).toDF("text"))

results = model.transform(data)
```

```scala
...
val puerile_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_puerile_clinical","en","clinical/models")
	.setInputCols(Array("token","chunk_embeddings"))
	.setOutputCol("resolution")
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, puerile_resolver))

val data = Seq("The patient is a 5-month-old infant who presented initially on Monday with a cold, cough, and runny nose for 2 days. Mom states she had no fever. Her appetite was good but she was spitting up a lot. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed a right TM, which was red. Left TM was okay. She was fairly congested but looked happy and playful. She was started on Amoxil and Aldex and we told to recheck in 2 weeks to recheck her ear. Mom returned to clinic again today because she got much worse overnight. She was having difficulty breathing. She was much more congested and her appetite had decreased significantly today. She also spiked a temperature yesterday of 102.6 and always having trouble sleeping secondary to congestion.").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

{:.h2_title}
## Results

```bash
   chunk                                    entity       icd10Puerile_description                          icd10Puerile_code

0  cold                                     Symptom_Name Prolonged pregnancy                               O481
1  cough                                    Symptom_Name Diseases of the respiratory system complicatin... O9953
2  runny nose                               Symptom_Name Other specified pregnancy related conditions, ... O26899
3  Mom                                      Gender       Classical hydatidiform mole                       O010
4  she                                      Gender       Other vomiting complicating pregnancy             O218
5  no                                       Negated      Continuing pregnancy after spontaneous abortio... O3111X0
6  fever                                    Symptom_Name Puerperal sepsis                                  O85
7  Her                                      Gender       Obesity complicating the puerperium               O99215
8  she                                      Gender       Other vomiting complicating pregnancy             O218
9  spitting up a lot                        Symptom_Name Eclampsia, unspecified as to time period          O159
10 She                                      Gender       Other vomiting complicating pregnancy             O218
11 no                                       Negated      Continuing pregnancy after spontaneous abortio... O3111X0
12 difficulty breathing                     Symptom_Name Other disorders of lactation                      O9279
13 her                                      Gender       Diseases of the nervous system complicating th... O99355
14 cough                                    Symptom_Name Diseases of the respiratory system complicatin... O9953
15 dry                                      Modifier     Cracked nipple associated with lactation          O9213
16 hacky                                    Modifier     Severe pre-eclampsia, unspecified trimester       O1410
17 She                                      Gender       Other vomiting complicating pregnancy             O218
18 fairly                                   Modifier     Maternal care for high head at term, not appli... O324XX0
19 congested                                Symptom_Name Postpartum thyroiditis                            O905
20 She                                      Gender       Other vomiting complicating pregnancy             O218
21 Amoxil                                   Drug_Name    Suppressed lactation                              O925
22 Aldex                                    Drug_Name    Severe pre-eclampsia, unspecified trimester       O1410
23 her                                      Gender       Diseases of the nervous system complicating th... O99355
24 Mom                                      Gender       Classical hydatidiform mole                       O010
25 she                                      Gender       Other vomiting complicating pregnancy             O218
26 She                                      Gender       Other vomiting complicating pregnancy             O218
27 difficulty breathing                     Symptom_Name Other disorders of lactation                      O9279
28 She                                      Gender       Other vomiting complicating pregnancy             O218
29 congested and her appetite had decreased Symptom_Name Decreased fetal movements, second trimester, n... O368120
30 She                                      Gender       Other vomiting complicating pregnancy             O218
31 102.6                                    Temperature  Postpartum coagulation defects                    O723
32 trouble sleeping                         Symptom_Name Other disorders of lactation                      O9279
33 secondary to                             Modifier     Unspecified pre-existing hypertension complica... O10919
34 congestion                               Symptom_Name Viral hepatitis complicating childbirth           O9842
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|---------------------------------------|
| Name:           | chunkresolve_icd10cm_puerile_clinical |
| Type:    | ChunkEntityResolverModel              |
| Compatibility:  | Spark NLP 2.4.5+                                 |
| License:        | Licensed                              |
|Edition:|Official|                            |
|Input labels:         | [token, chunk_embeddings]               |
|Output labels:        | [entity]                                |
| Language:       | en                                    |
| Case sensitive: | True                                  |
| Dependencies:  | embeddings_clinical                   |

{:.h2_title}
## Data Source
Trained on ICD10CM Dataset Range: O0000-O9989
https://www.icd10data.com/ICD10CM/Codes/O00-O9A
