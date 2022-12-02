---
layout: model
title: Sentence Entity Resolver for UMLS CUI Codes
author: John Snow Labs
name: sbiobertresolve_umls_major_concepts
date: 2021-05-16
tags: [entity_resolution, clinical, licensed, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.0.4
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts to 4 major categories of UMLS CUI codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It has faster load time, with a speedup of about 6X when compared to previous versions. Also the load process now is more memory friendly meaning that the maximum memory required during load time is smaller, reducing the chances of OOM exceptions, and thus relaxing hardware requirements.

## Predicted Entities

This model returns CUI (concept unique identifier) codes for `Clinical Findings`, `Medical Devices`, `Anatomical Structures` and `Injuries & Poisoning` terms

{:.btn-box}
[Live Demo](http://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_umls_major_concepts_en_3.0.4_3.0_1621188910976.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

```sbiobertresolve_umls_major_concepts``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_jsl``` as NER model. ```Cerebrovascular_Disease, Communicable_Disease, Diabetes, Disease_Syndrome_Disorder, Heart_Disease, Hyperlipidemia, Hypertension, Injury_or_Poisoning, Kidney_Disease, Medical-Device, Obesity, Oncological, Overweight, Psychological_Condition, Symptom, VS_Finding, ImagingFindings, EKG_Findings``` set in ```.setWhiteList()```.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")

resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_umls_major_concepts", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

pipeline = Pipeline(stages = [document_assembler, sentence_detector, tokens, embeddings, ner, ner_converter, chunk2doc, sbert_embedder, resolver])

data = spark.createDataFrame([["""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting."""]]).toDF("text")

results = pipeline.fit(data).transform(data)
```



{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.umls").predict("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting.""")
```

</div>

## Results

```bash
|    | ner_chunk                     | resolution   |
|---:|:------------------------------|:-------------|
|  0 | 28-year-old                   | C1864118     |
|  1 | female                        | C3887375     |
|  2 | gestational diabetes mellitus | C2183115     |
|  3 | eight years prior             | C5195266     |
|  4 | subsequent                    | C3844350     |
|  5 | type two diabetes mellitus    | C4014362     |
|  6 | T2DM                          | C4014362     |
|  7 | HTG-induced pancreatitis      | C4554179     |
|  8 | three years prior             | C1866782     |
|  9 | acute                         | C1332147     |
| 10 | hepatitis                     | C1963279     |
| 11 | obesity                       | C1963185     |
| 12 | body mass index               | C0578022     |
| 13 | 33.5 kg/m2                    | C2911054     |
| 14 | one-week                      | C0420331     |
| 15 | polyuria                      | C3278312     |
| 16 | polydipsia                    | C3278316     |
| 17 | poor appetite                 | C0541799     |
| 18 | vomiting                      | C1963281     |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_umls_major_concepts|
|Compatibility:|Healthcare NLP 3.0.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[umls_code]|
|Language:|en|
|Case sensitive:|false|