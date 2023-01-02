---
layout: model
title: Legal Relation Extraction (Whereas)
author: John Snow Labs
name: legre_whereas_md
date: 2022-11-09
tags: [en, legal, licensed, whereas, re]
task: Relation Extraction
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: RelationExtractionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_cuad_wheras_clause` Text Classifier to select only these paragraphs; 

This is a Relation Extraction model to infer relations between elements in WHEREAS clauses, more specifically the SUBJECT, the ACTION and the OBJECT. There are two relations possible: `has_subject` and `has_object`. You can also use `legpipe_whereas` which includes this model and its NER and also depedency parsing, to carry out chunk extraction using grammatical features (the dependency tree). This model requires `legner_whereas` as an NER in the pipeline. It's a `md` model with Unidirectional Relations, meaning that the model retrieves in chunk1 the left side of the relation (source), and in chunk2 the right side (target).

## Predicted Entities

`has_subject`, `has_object`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_whereas_md_en_1.0.0_3.0_1668013863138.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

documentAssembler = nlp.DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
  .setInputCols("document")\
  .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained('legner_whereas', 'en', 'legal/models')\
        .setInputCols(["document", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["document","token","ner"])\
        .setOutputCol("ner_chunk")

reDL = legal.RelationExtractionDLModel\
    .pretrained("legre_whereas_md", "en", "legal/models")\
    .setPredictionThreshold(0.9)\
    .setInputCols(["ner_chunk", "document"])\
    .setOutputCol("relations")
    
pipeline = Pipeline(stages=[
    documentAssembler,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter,
    reDL
])

empty_df = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_df)

text = """
Central Expressway, Suite 200, Dallas, TX 75080.

Background

The Supplier wishes to appoint the Distributor as its non-exclusive distributor for the promotion and sale of the Products within the Territory (both as defined below), and the Distributor wishes to promote and sell the Products within the Territory on the terms of this agreement.

Agreed terms

1. """

data = spark.createDataFrame([[text]]).toDF("text")
model = pipeline.fit(data)
res = model.transform(data)

```

</div>

## Results

```bash
+-----------+---------------+-------------+-----------+------------------------------------------------+---------------+-------------+-----------+------------------------------------------------+----------+
|relation   |entity1        |entity1_begin|entity1_end|chunk1                                          |entity2        |entity2_begin|entity2_end|chunk2                                          |confidence|
+-----------+---------------+-------------+-----------+------------------------------------------------+---------------+-------------+-----------+------------------------------------------------+----------+
|has_subject|WHEREAS_ACTION |76           |92         |wishes to appoint                               |WHEREAS_SUBJECT|63           |74         |The Supplier                                    |0.9994367 |
|has_subject|WHEREAS_OBJECT |94           |141        |the Distributor as its non-exclusive distributor|WHEREAS_SUBJECT|63           |74         |The Supplier                                    |0.92683166|
|has_subject|WHEREAS_SUBJECT|236          |250        |the Distributor                                 |WHEREAS_SUBJECT|63           |74         |The Supplier                                    |0.9829159 |
|has_object |WHEREAS_ACTION |76           |92         |wishes to appoint                               |WHEREAS_OBJECT |94           |141        |the Distributor as its non-exclusive distributor|0.900727  |
|has_object |WHEREAS_OBJECT |94           |141        |the Distributor as its non-exclusive distributor|WHEREAS_OBJECT |279          |290        |the Products                                    |0.96618503|
|has_subject|WHEREAS_ACTION |252          |268        |wishes to promote                               |WHEREAS_SUBJECT|236          |250        |the Distributor                                 |0.99969923|
+-----------+---------------+-------------+-----------+------------------------------------------------+---------------+-------------+-----------+------------------------------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_whereas_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|402.2 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
       label    Recall Precision        F1   Support
  has_object     0.974     0.991     0.983       116
 has_subject     0.977     0.986     0.981       213
       other     0.993     0.978     0.985       271
         Avg     0.981     0.985     0.983       -
Weighted-Avg     0.983     0.983     0.983       -
```
