---
layout: model
title: Sentence Entity Resolver for UMLS CUI Codes
author: John Snow Labs
name: sbiobertresolve_umls_findings
date: 2021-04-30
tags: [en, clinical, licensed, entity_resolution]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.0.2
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Map clinical entities to UMLS CUI codes.

## Predicted Entities

This model returns CUI (concept unique identifier) codes for 200K concepts from clinical findings.
https://www.nlm.nih.gov/research/umls/index.html

{:.btn-box}
[Live Demo](http://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_umls_findings_en_3.0.2_3.0_1619774838339.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")

resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_umls_findings","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

pipeline = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver])

data = spark.createDataFrame([["""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting."""]]).toDF("text")

results = pipeline.fit(data).transform(data)
```



{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.umls.findings").predict("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting.""")
```

</div>

## Results

```bash
|    | ner_chunk                             | cui_code   |
|---:|:--------------------------------------|:-----------|
|  0 | gestational diabetes mellitus         | C2183115   |
|  1 | subsequent type two diabetes mellitus | C3532488   |
|  2 | T2DM                                  | C3280267   |
|  3 | HTG-induced pancreatitis              | C4554179   |
|  4 | an acute hepatitis                    | C4750596   |
|  5 | obesity                               | C1963185   |
|  6 | a body mass index                     | C0578022   |
|  7 | polyuria                              | C3278312   |
|  8 | polydipsia                            | C3278316   |
|  9 | poor appetite                         | C0541799   |
| 10 | vomiting                              | C0042963   |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_umls_findings|
|Compatibility:|Healthcare NLP 3.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[umls_code]|
|Language:|en|

## Data Source

https://www.nlm.nih.gov/research/umls/index.html