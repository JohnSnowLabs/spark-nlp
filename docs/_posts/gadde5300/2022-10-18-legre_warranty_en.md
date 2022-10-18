---
layout: model
title: Legal Relation Extraction (Warranty)
author: John Snow Labs
name: legre_warranty
date: 2022-10-18
tags: [legal, en, re, licensed, warranty]
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

This is a Legal Relation Extraction Model to identify the Subject (who), Action (web), Object(the indemnification) and Indirect Object (to whom) from Warranty clauses.

## Predicted Entities

`is_warranty_indobject`, `is_warranty_object`, `is_warranty_subject`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_warranty_en_1.0.0_3.0_1666094112477.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = legal.NerModel.pretrained('legner_warranty', 'en', 'legal/models') \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")

ner_converter = nlp.NerConverter() \
        .setInputCols(["document","token","ner"]) \
        .setOutputCol("ner_chunk")

reDL = legal.RelationExtractionDLModel.pretrained("legre_warranty", "en", "legal/models") \
    .setPredictionThreshold(0.5) \
    .setInputCols(["ner_chunk", "document"]) \
    .setOutputCol("relations")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings, ner_model, ner_converter, reDL])
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_warranty|
|Compatibility:|Spark NLP for Legal 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|409.9 MB|

## References

In-house annotated examples from CUAD legal dataset