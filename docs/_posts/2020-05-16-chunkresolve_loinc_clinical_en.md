---
layout: model
title: ChunkResolver Loinc Clinical
author: John Snow Labs
name: chunkresolve_loinc_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-05-16
task: Entity Resolution
edition: Healthcare NLP 2.5.0
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
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance.

## Predicted Entities
LOINC Codes with ``clinical_embeddings``.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_LOINC/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_loinc_clinical_en_2.5.0_2.4_1589599195201.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/chunkresolve_loinc_clinical_en_2.5.0_2.4_1589599195201.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...    
loinc_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_loinc_clinical", "en", "clinical/models") \
  .setInputCols(["token", "chunk_embeddings"]) \
  .setOutputCol("loinc_code") \
  .setDistanceFunction("COSINE") \
  .setNeighbours(5)

pipeline_loinc = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, loinc_resolver])

data = spark.createDataFrame([["""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting."""]]).toDF("text")

model = pipeline_loinc.fit(data)

results = model.transform(data)
```

```scala
...
val loinc_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_loinc_clinical", "en", "clinical/models")
  .setInputCols(Array("token", "chunk_embeddings"))
  .setOutputCol("loinc_code")
  .setDistanceFunction("COSINE")
  .setNeighbours(5)

val pipeline_loinc = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, loinc_resolver))

val data = Seq("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting.").toDF("text")

val result = pipeline_loinc.fit(data).transform(data)
```
</div>

{:.h2_title}
## Results

```bash
                                      Chunk  loinc-Code

0             gestational diabetes mellitus  44877-9
1                type two diabetes mellitus  44877-9
2                                      T2DM  93692-2
3 prior episode of HTG-induced pancreatitis  85695-5
4        associated with an acute hepatitis  24363-4
5            obesity with a body mass index  47278-7
6                        BMI) of 33.5 kg/m2  47214-2
7                                  polyuria  35234-4
8                                polydipsia  25541-4
9                             poor appetite  50056-1
10                                 vomiting  34175-0
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|-----------------------------|
| Name:           | chunkresolve_loinc_clinical |
| Type:    | ChunkEntityResolverModel    |
| Compatibility:  | Spark NLP 2.5.0+                       |
| License:        | Licensed                    |
|Edition:|Official|                  |
|Input labels:         | [token, chunk_embeddings]     |
|Output labels:        | [entity]                      |
| Language:       | en                          |
| Case sensitive: | True                        |
| Dependencies:  | embeddings_clinical         |

{:.h2_title}
## Data Source
Trained on LOINC dataset with ``embeddings_clinical``.
https://loinc.org/
