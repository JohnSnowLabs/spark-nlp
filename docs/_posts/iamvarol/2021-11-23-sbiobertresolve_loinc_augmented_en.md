---
layout: model
title: Sentence Entity Resolver for LOINC (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_loinc_augmented
date: 2021-11-23
tags: [loinc, entity_resolution, clinical, en, licensed]
task: Entity Resolution
language: en
edition: Spark NLP for Healthcare 3.3.2
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted clinical NER entities to LOINC codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It trained on the augmented version of the dataset which is used in previous LOINC resolver models.

## Predicted Entities



{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_loinc_augmented_en_3.3.2_2.4_1637664939262.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
     .pretrained("sbluebert_base_uncased_mli","en","clinical/models")\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")

resolver = SentenceEntityResolverModel.pretrained("sbluebertresolve_loinc_augmented","en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")

pipeline_loinc = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver])

model = pipeline_loinc.fit(spark.createDataFrame([["""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting."""]]).toDF("text"))

results = model.transform(data)
```
```scala
val documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
      .setInputCols("ner_chunk")\
      .setOutputCol("sbert_embeddings")
    
val loinc_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_loinc_augmented", "en", "clinical/models") \
      .setInputCols(Array("ner_chunk", "sbert_embeddings")) \
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

val loinc_pipelineModel = new PipelineModel().setStages(Array(documentAssembler, sbert_embedder, loinc_resolver))

val data = Seq("She is given Fragmin 5000 units subcutaneously daily, Xenaderm to wounds topically b.i.d., OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, 
levothyroxine 0.1 mg p.o. daily, Prevacid 30 mg daily, Avandia 4 mg daily, aspirin 81 mg daily, Neurontin 400 mg p.o. t.i.d., 
Percocet 5/325 mg 2 tablets q.4 h. p.r.n., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin, Wellbutrin 100 mg p.o. daily.").toDF("text")

val result = pipeline.fit(data).transform(data)
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
|Model Name:|sbiobertresolve_loinc_augmented|
|Compatibility:|Spark NLP for Healthcare 3.3.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[loinc_code]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on standard LOINC coding system.