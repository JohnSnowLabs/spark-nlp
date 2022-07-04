---
layout: model
title: Loinc Sentence Entity Resolver
author: John Snow Labs
name: sbiobertresolve_loinc
date: 2021-05-16
tags: [entity_resolution, clinical, licensed, en]
task: Entity Resolution
language: en
edition: Spark NLP for Healthcare 3.0.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted clinical NER entities to LOINC codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings, and has faster load time, with a speedup of about 6X when compared to previous versions. Also the load process now is more memory friendly meaning that the maximum memory required during load time is smaller, reducing the chances of OOM exceptions, and thus relaxing hardware requirements.

## Predicted Entities

Predicts LOINC Codes and their normalized definition for each chunk.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_LOINC/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_loinc_en_3.0.4_3.0_1621189494152.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



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
     .pretrained("sbiobertresolve_loinc","en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")

pipeline_loinc = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver])

data = spark.createDataFrame([["""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting."""]]).toDF("text")

results = pipeline_loinc.fit(data).transform(data)
```



{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.loinc").predict("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting.""")
```

</div>

## Results

```bash
|    | chunk                                 | loinc_code   |
|---:|:--------------------------------------|:-------------|
|  0 | gestational diabetes mellitus         | 45636-8      |
|  1 | subsequent type two diabetes mellitus | 44877-9      |
|  2 | T2DM                                  | 45636-8      |
|  3 | HTG-induced pancreatitis              | 66667-7      |
|  4 | an acute hepatitis                    | 45690-5      |
|  5 | obesity                               | 73708-0      |
|  6 | a body mass index                     | 59574-4      |
|  7 | BMI                                   | 59574-4      |
|  8 | polyuria                              | 28239-2      |
|  9 | polydipsia                            | 90552-1      |
| 10 | poor appetite                         | 28387-9      |
| 11 | vomiting                              | 81224-8      |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_loinc|
|Compatibility:|Spark NLP for Healthcare 3.0.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[loinc_code]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on standard LOINC coding system.