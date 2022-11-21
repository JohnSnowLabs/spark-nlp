---
layout: model
title: Sentence Entity Resolver for UMLS CUI Codes (Disease or Syndrome)
author: John Snow Labs
name: sbiobertresolve_umls_disease_syndrome
date: 2021-10-11
tags: [entity_resolution, licensed, clinical, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.2.3
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities to UMLS CUI codes. It is trained on `2021AB` UMLS dataset. The complete dataset has 127 different categories, and this model is trained on the `Disease or Syndrome` category using `sbiobert_base_cased_mli` embeddings.

## Predicted Entities

`Predicts UMLS codes for Diseases & Syndromes medical concepts`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_umls_disease_syndrome_en_3.2.3_3.0_1633911418710.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use


```sbiobertresolve_umls_disease_syndrome``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_jsl``` as NER model. ```Cerebrovascular_Disease, Communicable_Disease, Diabetes,Disease_Syndrome_Disorder, Heart_Disease, Hyperlipidemia, Hypertension,Injury_or_Poisoning, Kidney_Disease, Obesity, Oncological, Overweight, Psychological_Condition, Symptom, VS_Finding, ImagingFindings, EKG_Findings``` set in ```.setWhiteList()```.


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")

resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_umls_disease_syndrome","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

pipeline = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver])

data = spark.createDataFrame([["""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting."""]]).toDF("text")

results = pipeline.fit(data).transform(data)
```
```scala
...
val chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sbert_embeddings")

val resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_umls_disease_syndrome", "en", "clinical/models") 
.setInputCols(Array("ner_chunk_doc", "sbert_embeddings")) 
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val p_model = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver))

val data = Seq("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting.").toDF("text") 

val res = p_model.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.umls_disease_syndrome").predict("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting.""")
```

</div>

## Results

```bash
|    | chunk                                 | code     | code_description                      | all_k_code_desc                                              | all_k_codes                                                                                                                                                                                              |
|---:|:--------------------------------------|:---------|:--------------------------------------|:-------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | gestational diabetes mellitus         | C0085207 | gestational diabetes mellitus         | ['C0085207', 'C0032969', 'C2063017', 'C1283034', 'C0271663'] | ['gestational diabetes mellitus', 'pregnancy diabetes mellitus', 'pregnancy complicated by diabetes mellitus', 'maternal diabetes mellitus', 'gestational diabetes mellitus, a2']                        |
|  1 | subsequent type two diabetes mellitus | C0348921 | pre-existing type 2 diabetes mellitus | ['C0348921', 'C1719939', 'C0011860', 'C0877302', 'C0271640'] | ['pre-existing type 2 diabetes mellitus', 'disorder associated with type 2 diabetes mellitus', 'diabetes mellitus, type 2', 'insulin-requiring type 2 diabetes mellitus', 'secondary diabetes mellitus'] |
|  2 | HTG-induced pancreatitis              | C0376670 | alcohol-induced pancreatitis          | ['C0376670', 'C1868971', 'C4302243', 'C0267940', 'C2350449'] | ['alcohol-induced pancreatitis', 'toxic pancreatitis', 'igg4-related pancreatitis', 'hemorrhage pancreatitis', 'graft pancreatitis']                                                                     |
|  3 | an acute hepatitis                    | C0019159 | acute hepatitis                       | ['C0019159', 'C0276434', 'C0267797', 'C1386146', 'C2063407'] | ['acute hepatitis a', 'acute hepatitis a', 'acute hepatitis', 'acute infectious hepatitis', 'acute hepatitis e']                                                                                         |
|  4 | obesity                               | C0028754 | obesity                               | ['C0028754', 'C0342940', 'C0342942', 'C0857116', 'C1561826'] | ['obesity', 'abdominal obesity', 'generalized obesity', 'obesity gross', 'overweight and obesity']                                                                                                       |
|  5 | polyuria                              | C0018965 | hematuria                             | ['C0018965', 'C0151582', 'C3888890', 'C0268556', 'C2936921'] | ['hematuria', 'uricosuria', 'polyuria-polydipsia syndrome', 'saccharopinuria', 'saccharopinuria']                                                                                                        |
|  6 | polydipsia                            | C0268813 | primary polydipsia                    | ['C0268813', 'C0030508', 'C3888890', 'C0393777', 'C0206085'] | ['primary polydipsia', 'parasomnia', 'polyuria-polydipsia syndrome', 'hypnogenic paroxysmal dystonias', 'periodic hypersomnias']                                                                         |
|  7 | poor appetite                         | C0003123 | lack of appetite                      | ['C0003123', 'C0011168', 'C0162429', 'C1282895', 'C0039338'] | ['lack of appetite', 'poor swallowing', 'poor nutrition', 'neurologic unpleasant taste', 'taste dis']                                                                                                    |
|  8 | vomiting                              | C0152164 | periodic vomiting                     | ['C0152164', 'C0267172', 'C0152517', 'C0011119', 'C0152227'] | ['periodic vomiting', 'habit vomiting', 'viral vomiting', 'choking', 'tearing']                                                                                                                          |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_umls_disease_syndrome|
|Compatibility:|Healthcare NLP 3.2.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_chunk_embeddings]|
|Output Labels:|[output]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on `2021AB` UMLS dataset's ' Disease or Syndrome` category. https://www.nlm.nih.gov/research/umls/index.html