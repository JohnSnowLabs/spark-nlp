---
layout: model
title: Legal Relation Extraction (Whereas, sm, Bidirectional))
author: John Snow Labs
name: legre_whereas
date: 2022-08-24
tags: [en, legal, re, relations, licensed]
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
- Use the `legclf_cuad_whereas_clause` Text Classifier to select only these paragraphs; 

This is a Relation Extraction model to infer relations between elements in WHEREAS clauses, more specifically the SUBJECT, the ACTION and the OBJECT. There are two relations possible: `has_subject` and `has_object`.

You can also use `legpipe_whereas` which includes this model and its NER and also depedency parsing, to carry out chunk extraction using grammatical features (the dependency tree).

This model is a `sm` model without meaningful directions in the relations (the model was not trained to understand if the direction of the relation is from left to right or right to left).

There are bigger models in Models Hub trained also with directed relationships.

## Predicted Entities

`has_subject`, `has_object`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_whereas_en_1.0.0_3.2_1661341573628.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legre_whereas_en_1.0.0_3.2_1661341573628.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    .pretrained("legre_whereas", "en", "legal/models")\
    .setPredictionThreshold(0.5)\
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

text = """
WHEREAS VerticalNet owns and operates a series of online communities ( as defined below ) that are accessible via the world wide web , each of which is designed to be an online gathering place for businesses of a certain type or within a certain industry ;
"""

data = spark.createDataFrame([[text]]).toDF("text")
model = pipeline.fit(data)
res = model.transform(data)
```

</div>

## Results

```bash
   relation         entity1 entity1_begin entity1_end      chunk1        entity2 entity2_begin entity2_end                         chunk2 confidence
has_subject WHEREAS_SUBJECT            11          21 VerticalNet WHEREAS_ACTION            32          39                       operates  0.9982886
has_subject WHEREAS_SUBJECT            11          21 VerticalNet WHEREAS_OBJECT            41          70 a series of online communities  0.9890683
 has_object  WHEREAS_ACTION            32          39    operates WHEREAS_OBJECT            41          70 a series of online communities  0.7831568
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_whereas|
|Type:|legal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|409.9 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
label               Recall    Precision  F1          Support
has_object          0.946     0.981      0.964        56
has_subject         0.952     0.988      0.969        83
no_rel              1.000     0.970      0.985       161
Avg.                0.966     0.980      0.973        -
Weighted-Avg.       0.977     0.977      0.977        -
```
