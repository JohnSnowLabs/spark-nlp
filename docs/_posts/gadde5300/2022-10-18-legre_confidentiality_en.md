---
layout: model
title: Legal Relation Extraction (Confidentiality)
author: John Snow Labs
name: legre_confidentiality
date: 2022-10-18
tags: [legal, en, re, licensed, is_confidentiality_subject]
task: Relation Extraction
language: en
edition: Spark NLP for Legal 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Legal Relation Extraction Model to identify the Subject (who), Action (web), Object(the indemnification) and Indirect Object (to whom) from confidentiality clauses.

## Predicted Entities

`is_confidentiality_indobject`, `is_confidentiality_object`, `is_confidentiality_subject`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_confidentiality_en_1.0.0_3.0_1666095682357.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

reDL = legal.RelationExtractionDLModel.pretrained("legre_confidentiality", "en", "legal/models") \
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
|relation                    |entity1                        |entity1_begin|entity1_end|chunk1                  |entity2                        |entity2_begin|entity2_end|chunk2                  |confidence|
|----------------------------|-------------------------------|-------------|-----------|------------------------|-------------------------------|-------------|-----------|------------------------|----------|
|is_confidentiality_subject  |CONFIDENTIALITY_SUBJECT        |1            |10         |Each party              |CONFIDENTIALITY_ACTION         |12           |31         |will promptly return    |0.829208  |
|is_confidentiality_indobject|CONFIDENTIALITY_SUBJECT        |1            |10         |Each party              |CONFIDENTIALITY_INDIRECT_OBJECT|40           |44         |other                   |0.9989385 |
|is_confidentiality_object   |CONFIDENTIALITY_SUBJECT        |1            |10         |Each party              |CONFIDENTIALITY                |63           |86         |Confidential Information|0.9772866 |
|is_confidentiality_indobject|CONFIDENTIALITY_SUBJECT        |1            |10         |Each party              |CONFIDENTIALITY_INDIRECT_OBJECT|95           |105        |other party             |0.9970458 |
|is_confidentiality_indobject|CONFIDENTIALITY_ACTION         |12           |31         |will promptly return    |CONFIDENTIALITY_INDIRECT_OBJECT|40           |44         |other                   |0.99961615|
|is_confidentiality_object   |CONFIDENTIALITY_ACTION         |12           |31         |will promptly return    |CONFIDENTIALITY                |63           |86         |Confidential Information|0.99497294|
|is_confidentiality_indobject|CONFIDENTIALITY_ACTION         |12           |31         |will promptly return    |CONFIDENTIALITY_INDIRECT_OBJECT|95           |105        |other party             |0.9985304 |
|is_confidentiality_object   |CONFIDENTIALITY_INDIRECT_OBJECT|40           |44         |other                   |CONFIDENTIALITY                |63           |86         |Confidential Information|0.93832284|
|is_confidentiality_indobject|CONFIDENTIALITY_INDIRECT_OBJECT|40           |44         |other                   |CONFIDENTIALITY_INDIRECT_OBJECT|95           |105        |other party             |0.99965453|
|is_confidentiality_indobject|CONFIDENTIALITY                |63           |86         |Confidential Information|CONFIDENTIALITY_INDIRECT_OBJECT|95           |105        |other party             |0.9798117 |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_confidentiality|
|Compatibility:|Spark NLP for Legal 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|409.9 MB|

## References

In-house annotated examples from CUAD legal dataset


## Benchmarking

```bash
Relation                        Recall    Precision     F1   Support

is_confidentiality_indobject    1.000     1.000      1.000        28
is_confidentiality_object       1.000     0.981      0.990        51
is_confidentiality_subject      0.970     1.000      0.985        33

Avg.                            0.990     0.994     0.992

Weighted Avg.                   0.991     0.991     0.991
```
