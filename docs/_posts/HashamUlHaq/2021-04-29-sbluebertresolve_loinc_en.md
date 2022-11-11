---
layout: model
title: Loinc Sentence Entity Resolver
author: John Snow Labs
name: sbluebertresolve_loinc
date: 2021-04-29
tags: [en, licensed, clinical, entity_resolution]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
deprecated: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Map clinical NER entities to LOINC codes.

## Predicted Entities

LOINC codes - per input NER entity

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbluebertresolve_loinc_en_3.0.2_3.0_1619678534366.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
     .pretrained("sbluebert_base_uncased_mli","en","clinical/models")\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")

resolver = SentenceEntityResolverModel.pretrained("sbluebertresolve_loinc","en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")

pipeline_loinc = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver])

model = pipeline_loinc.fit(spark.createDataFrame([["""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting."""]]).toDF("text"))

results = model.transform(data)
```

</div>

## Results

```bash
|    | chunk                                 | loinc_code   |
|---:|:--------------------------------------|:-------------|
|  0 | gestational diabetes mellitus         | 45636-8      |
|  1 | subsequent type two diabetes mellitus | 44877-9      |
|  2 | T2DM                                  | 45636-8      |
|  3 | HTG-induced pancreatitis              | 79102-0      |
|  4 | an acute hepatitis                    | 28083-4      |
|  5 | obesity                               | 50227-8      |
|  6 | a body mass index                     | 59574-4      |
|  7 | BMI                                   | 59574-4      |
|  8 | polyuria                              | 28239-2      |
|  9 | polydipsia                            | 90552-1      |
| 10 | poor appetite                         | 65961-5      |
| 11 | vomiting                              | 81224-8      |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbluebertresolve_loinc|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[loinc_code]|
|Language:|en|

## Data Source

Trained on standard LOINC coding system.
