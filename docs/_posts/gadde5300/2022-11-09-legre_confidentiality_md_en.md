---
layout: model
title: Legal Relation Extraction (Confidentiality, md, Unidirectional)
author: John Snow Labs
name: legre_confidentiality_md
date: 2022-11-09
tags: [en, legal, licensed, confidentiality, re]
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

This is a Legal Relation Extraction Model to identify the Subject (who), Action (what), Object(the confidentiality) and Indirect Object (to whom) from confidentiality clauses. This model requires `legner_confidentiality` as an NER in the pipeline. It's a `md` model with Unidirectional Relations, meaning that the model retrieves in chunk1 the left side of the relation (source), and in chunk2 the right side (target).

## Predicted Entities

`is_confidentiality_indobject`, `is_confidentiality_object`, `is_confidentiality_subject`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_confidentiality_md_en_1.0.0_3.0_1668006317769.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
+----------------------------+-------------------------------+-------------+-----------+--------------------+-------------------------------+-------------+-----------+------------------------+----------+
|relation                    |entity1                        |entity1_begin|entity1_end|chunk1              |entity2                        |entity2_begin|entity2_end|chunk2                  |confidence|
+----------------------------+-------------------------------+-------------+-----------+--------------------+-------------------------------+-------------+-----------+------------------------+----------+
|is_confidentiality_object   |CONFIDENTIALITY_SUBJECT        |0            |9          |Each party          |CONFIDENTIALITY_ACTION         |11           |30         |will promptly return    |0.6433299 |
|is_confidentiality_indobject|CONFIDENTIALITY_SUBJECT        |0            |9          |Each party          |CONFIDENTIALITY_INDIRECT_OBJECT|39           |43         |other                   |0.5328208 |
|is_confidentiality_object   |CONFIDENTIALITY_SUBJECT        |0            |9          |Each party          |CONFIDENTIALITY                |62           |85         |Confidential Information|0.9985228 |
|is_confidentiality_indobject|CONFIDENTIALITY_ACTION         |11           |30         |will promptly return|CONFIDENTIALITY_INDIRECT_OBJECT|39           |43         |other                   |0.97346765|
|is_confidentiality_object   |CONFIDENTIALITY_ACTION         |11           |30         |will promptly return|CONFIDENTIALITY                |62           |85         |Confidential Information|0.9521258 |
|is_confidentiality_object   |CONFIDENTIALITY_INDIRECT_OBJECT|39           |43         |other               |CONFIDENTIALITY                |62           |85         |Confidential Information|0.91878176|
+----------------------------+-------------------------------+-------------+-----------+--------------------+-------------------------------+-------------+-----------+------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_confidentiality_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|402.3 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
Relation                            Recall    Precision  F1          Support 

is_confidentiality_indobject         0.960     1.000     0.980        25 
is_confidentiality_object            1.000     0.933     0.966        56
is_confidentiality_subject           0.935     1.000     0.967        31
other                                0.989     1.000     0.994        88

Avg.                                 0.971     0.983     0.977

Weighted Avg.                        0.980     0.981     0.980
```
