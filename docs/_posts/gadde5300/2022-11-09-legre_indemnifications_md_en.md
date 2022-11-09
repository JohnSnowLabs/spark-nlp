---
layout: model
title: Legal Relation Extraction (Indemnifications, md, Unidirectional)
author: John Snow Labs
name: legre_indemnifications_md
date: 2022-11-09
tags: [en, legal, licensed, indemnifications, re]
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

This is a Relation Extraction model to group the different entities extracted with the Indemnification NER model (see `legner_bert_indemnifications` in Models Hub). This model requires `legner_bert_indemnifications` as an NER in the pipeline. It's a `md` model with Unidirectional Relations, meaning that the model retrieves in chunk1 the left side of the relation (source), and in chunk2 the right side (target).

## Predicted Entities

`is_indemnification_subject`, `is_indemnification_object`, `is_indemnification_indobject`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_indemnifications_md_en_1.0.0_3.0_1668012403648.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

reDL = legal.RelationExtractionDLModel.pretrained("legre_indemnifications_md", "en", "legal/models") \
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
+----------------------------+-----------------------+-------------+-----------+---------------+-------------------------------+-------------+-----------+-----------+----------+
|relation                    |entity1                |entity1_begin|entity1_end|chunk1         |entity2                        |entity2_begin|entity2_end|chunk2     |confidence|
+----------------------------+-----------------------+-------------+-----------+---------------+-------------------------------+-------------+-----------+-----------+----------+
|is_indemnification_subject  |INDEMNIFICATION_ACTION |12           |26         |shall indemnify|INDEMNIFICATION_SUBJECT        |4            |10         |Company    |0.9905861 |
|is_indemnification_subject  |INDEMNIFICATION_ACTION |32           |44         |hold harmless  |INDEMNIFICATION_SUBJECT        |4            |10         |Company    |0.9996145 |
|is_indemnification_indobject|INDEMNIFICATION_SUBJECT|4            |10         |Company        |INDEMNIFICATION_INDIRECT_OBJECT|46           |48         |HOC        |0.9948344 |
|is_indemnification_object   |INDEMNIFICATION_ACTION |12           |26         |shall indemnify|INDEMNIFICATION                |58           |67         |any losses |0.9983841 |
|is_indemnification_object   |INDEMNIFICATION_ACTION |12           |26         |shall indemnify|INDEMNIFICATION                |70           |75         |claims     |0.9972869 |
|is_indemnification_object   |INDEMNIFICATION_ACTION |12           |26         |shall indemnify|INDEMNIFICATION                |78           |84         |damages    |0.99586076|
|is_indemnification_object   |INDEMNIFICATION_ACTION |12           |26         |shall indemnify|INDEMNIFICATION                |89           |99         |liabilities|0.9969894 |
|is_indemnification_object   |INDEMNIFICATION_ACTION |32           |44         |hold harmless  |INDEMNIFICATION                |58           |67         |any losses |0.9989536 |
|is_indemnification_object   |INDEMNIFICATION_ACTION |32           |44         |hold harmless  |INDEMNIFICATION                |70           |75         |claims     |0.99755704|
|is_indemnification_object   |INDEMNIFICATION_ACTION |32           |44         |hold harmless  |INDEMNIFICATION                |78           |84         |damages    |0.99725854|
|is_indemnification_object   |INDEMNIFICATION_ACTION |32           |44         |hold harmless  |INDEMNIFICATION                |89           |99         |liabilities|0.997675  |
+----------------------------+-----------------------+-------------+-----------+---------------+-------------------------------+-------------+-----------+-----------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_indemnifications_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|402.3 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash

Relation                        Recall    Precision  F1       Support

is_indemnification_indobject     1.000     1.000     1.000        18
is_indemnification_object        0.972     1.000     0.986        36
is_indemnification_subject       0.800     0.800     0.800        10
other                            0.972     0.946     0.959        36

Avg.                0.936     0.936     0.936

Weighted Avg.       0.960     0.961     0.960

```