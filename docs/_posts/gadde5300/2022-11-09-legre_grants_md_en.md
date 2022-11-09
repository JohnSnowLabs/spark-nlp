---
layout: model
title: Legal Relation Extraction (Grants, md, Unidirectional)
author: John Snow Labs
name: legre_grants_md
date: 2022-11-09
tags: [en, legal, licensed, grants, re]
task: Relation Extraction
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model requires `legner_bert_grants` as an NER in the pipeline. It's a `md` model with Unidirectional Relations, meaning that the model retrieves in chunk1 the left side of the relation (source), and in chunk2 the right side (target).

## Predicted Entities

`allows`, `is_allowed_to`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_grants_md_en_1.0.0_3.0_1668017439874.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = legal.NerModel.pretrained('legner_confidentiality', 'en', 'legal/models') \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")

ner_converter = nlp.NerConverter() \
        .setInputCols(["document","token","ner"]) \
        .setOutputCol("ner_chunk")

reDL = legal.RelationExtractionDLModel.pretrained("legre_confidentiality_md", "en", "legal/models") \
    .setPredictionThreshold(0.5) \
    .setInputCols(["ner_chunk", "document"]) \
    .setOutputCol("relations")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings, ner_model, ner_converter, reDL])

text = "Each party will promptly return to the other upon request any Confidential Information of the other party then in its possession or under its control."

data = spark.createDataFrame([[text]]).toDF("text")
model = pipeline.fit(data)
res = model.transform(data)
```

</div>

## Results

```bash
+-------------+--------------------------+-------------+-----------+----------------------------------------------------------------------------+--------------------------+-------------+-----------+----------------------------------------------------------------------------+----------+
|relation     |entity1                   |entity1_begin|entity1_end|chunk1                                                                      |entity2                   |entity2_begin|entity2_end|chunk2                                                                      |confidence|
+-------------+--------------------------+-------------+-----------+----------------------------------------------------------------------------+--------------------------+-------------+-----------+----------------------------------------------------------------------------+----------+
|allows       |PERMISSION_SUBJECT        |92           |101        |Diversinet                                                                  |PERMISSION_INDIRECT_OBJECT|120          |127        |Reseller                                                                    |0.99999297|
|is_allowed_to|PERMISSION_SUBJECT        |92           |101        |Diversinet                                                                  |PERMISSION                |132          |145        |exclusive, non                                                              |0.97158235|
|is_allowed_to|PERMISSION_INDIRECT_OBJECT|120          |127        |Reseller                                                                    |PERMISSION                |132          |145        |exclusive, non                                                              |0.9999945 |
|is_allowed_to|PERMISSION_INDIRECT_OBJECT|120          |127        |Reseller                                                                    |PERMISSION                |148          |223        |transferable and non-assignable right to market, sell, and sub-license those|0.99987125|
|is_allowed_to|PERMISSION                |148          |223        |transferable and non-assignable right to market, sell, and sub-license those|PERMISSION                |132          |145        |exclusive, non                                                              |0.9956748 |
+-------------+--------------------------+-------------+-----------+----------------------------------------------------------------------------+--------------------------+-------------+-----------+----------------------------------------------------------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_grants_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|402.2 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash

Relation           Recall Precision        F1   Support

allows              1.000     1.000     1.000        32
is_allowed_to       1.000     1.000     1.000        36
other               1.000     1.000     1.000        32

Avg.                1.000     1.000     1.000

Weighted Avg.       1.000     1.000     1.000

```