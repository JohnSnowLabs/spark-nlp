---
layout: model
title: Legal Relation Extraction (Warranty)
author: John Snow Labs
name: legre_warranty
date: 2022-10-19
tags: [legal, en, re, licensed, warranty]
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
- Use the `legclf_warranty_clause` Text Classifier to select only these paragraphs; 

This is a Legal Relation Extraction Model to identify the Subject (who), Action (web), Object(the indemnification) and Indirect Object (to whom) from warranty clauses.

## Predicted Entities

`is_warranty_indobject`, `is_warranty_object`, `is_warranty_subject`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_warranty_en_1.0.0_3.0_1666154293071.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

text = """ARTICLE XI - WARRANTIES   11.1 In addition to the warranties set forth in Article IX of the General Terms and Conditions of Transporter's FERC Gas Tariff, Shipper warrants the following:   (a) Shipper warrants that all upstream and downstream transportation arrangements are in place, or will be in place as of the requested effective date of service, and that it has advised the upstream and downstream transporters of the receipt and delivery points under this Agreement and any quantity limitations for each point as specified on Exhibit "A" attached hereto."""

data = spark.createDataFrame([[text]]).toDF("text")
model = pipeline.fit(data)
res = model.transform(data)
```

</div>

## Results

```bash
|relation           |entity1         |entity1_begin|entity1_end|chunk1  |entity2        |entity2_begin|entity2_end|chunk2                                                                                                                                 |confidence|
|-------------------|----------------|-------------|-----------|--------|---------------|-------------|-----------|---------------------------------------------------------------------------------------------------------------------------------------|----------|
|is_warranty_subject|WARRANTY_SUBJECT|158          |164        |Shipper |WARRANTY_ACTION|166          |173        |warrants                                                                                                                               |0.98402506|
|is_warranty_subject|WARRANTY_SUBJECT|196          |202        |Shipper |WARRANTY_ACTION|204          |211        |warrants                                                                                                                               |0.9707028 |
|is_warranty_object |WARRANTY_SUBJECT|196          |202        |Shipper |WARRANTY       |218          |352        |all upstream and downstream transportation arrangements are in place, or will be in place as of the requested effective date of service|0.9917001 |
|is_warranty_object |WARRANTY_SUBJECT|196          |202        |Shipper |WARRANTY       |367          |474        |has advised the upstream and downstream transporters of the receipt and delivery points under this Agreement                           |0.79867786|
|is_warranty_object |WARRANTY_ACTION |204          |211        |warrants|WARRANTY       |218          |352        |all upstream and downstream transportation arrangements are in place, or will be in place as of the requested effective date of service|0.97821265|
|is_warranty_object |WARRANTY_ACTION |204          |211        |warrants|WARRANTY       |367          |474        |has advised the upstream and downstream transporters of the receipt and delivery points under this Agreement                           |0.80337876|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_warranty|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|409.9 MB|

## References

In-house annotated examples from CUAD legal dataset

## Benchmarking

```bash
                label   Recall Precision        F1   Support
is_warranty_indobject    1.000     1.000     1.000        15
   is_warranty_object    1.000     1.000     1.000        44
  is_warranty_subject    1.000     1.000     1.000        29
                  Avg    1.000     1.000     1.000        -
         Weighted-Avg    1.000     1.000     1.000        -
```
