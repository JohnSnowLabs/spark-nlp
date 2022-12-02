---
layout: model
title: Legal Relation Extraction (Parties, Alias, Dates, Document Type, Sm, Bidirectional)
author: John Snow Labs
name: legre_contract_doc_parties
date: 2022-08-12
tags: [en, legal, re, relations, agreements, licensed]
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
IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_introduction_clause` Text Classifier to select only these paragraphs; 

This is a Legal Relation Extraction model, which can be used after the NER Model for extracting Parties, Document Types, Effective Dates and Aliases, called `legner_contract_doc_parties`.

As an output, you will get the relations linking the different concepts together, if such relation exists. The list of relations is:

- dated_as: A Document has an Effective Date
- has_alias: The Alias of a Party all along the document
- has_collective_alias: An Alias hold by several parties at the same time
- signed_by: Between a Party and the document they signed

This model is a `sm` model without meaningful directions in the relations (the model was not trained to understand if the direction of the relation is from left to right or right to left). There are bigger models in Models Hub trained also with directed relationships.


## Predicted Entities

`dated_as`, `has_alias`, `has_collective_alias`, `signed_by`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/LEGALRE_PARTIES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_contract_doc_parties_en_1.0.0_3.2_1660293010932.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = nlp.BertEmbeddings.pretrained("bert_base_uncased_legal", "en") \
    .setInputCols("document", "token") \
    .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained('legner_contract_doc_parties', 'en', 'legal/models')\
    .setInputCols(["document", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["document","token","ner"])\
    .setOutputCol("ner_chunk")

reDL = legal.RelationExtractionDLModel().pretrained('legre_contract_doc_parties', 'en', 'legal/models')\
    .setPredictionThreshold(0.5)\
    .setInputCols(["ner_chunk", "document"])\
    .setOutputCol("relations")

nlpPipeline = Pipeline(stages=[
    documentAssembler,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter,
    reDL
])
    
text='''
This INTELLECTUAL PROPERTY AGREEMENT (this "Agreement"), dated as of December 31, 2018 (the "Effective Date") is entered into by and between Armstrong Flooring, Inc., a Delaware corporation ("Seller") and AFI Licensing LLC, a Delaware limited liability company ("Licensing" and together with Seller, "Arizona") and AHF Holding, Inc. (formerly known as Tarzan HoldCo, Inc.), a Delaware corporation ("Buyer") and Armstrong Hardwood Flooring Company, a Tennessee corporation (the "Company" and together with Buyer the "Buyer Entities") (each of Arizona on the one hand and the Buyer Entities on the other hand, a "Party" and collectively, the "Parties").
'''

data = spark.createDataFrame([[text]]).toDF("text")
model = nlpPipeline.fit(data)
```

</div>

## Results

```bash
relation entity1 entity1_begin entity1_end                              chunk1 entity2 entity2_begin entity2_end                  chunk2 confidence
            dated_as     DOC             6          36     INTELLECTUAL PROPERTY AGREEMENT EFFDATE            70          86       December 31, 2018  0.9933402
           signed_by     DOC             6          36     INTELLECTUAL PROPERTY AGREEMENT   PARTY           142         164 Armstrong Flooring, Inc  0.6235637
           signed_by     DOC             6          36     INTELLECTUAL PROPERTY AGREEMENT   PARTY           316         331        AHF Holding, Inc  0.5001139
           has_alias   PARTY           142         164             Armstrong Flooring, Inc   ALIAS           193         198                  Seller 0.93385726
           has_alias   PARTY           206         222                   AFI Licensing LLC   ALIAS           264         272               Licensing  0.9859913
has_collective_alias   ALIAS           293         298                              Seller   ALIAS           302         308                 Arizona 0.82137156
           has_alias   PARTY           316         331                    AHF Holding, Inc   ALIAS           400         404                   Buyer  0.8178999
           has_alias   PARTY           412         446 Armstrong Hardwood Flooring Company   ALIAS           479         485                 Company  0.9557921
           has_alias   PARTY           412         446 Armstrong Hardwood Flooring Company   ALIAS           575         579                   Buyer  0.6778585
           has_alias   PARTY           412         446 Armstrong Hardwood Flooring Company   ALIAS           612         616                   Party  0.6778583
           has_alias   PARTY           412         446 Armstrong Hardwood Flooring Company   ALIAS           642         648                 Parties  0.6778585
has_collective_alias   ALIAS           506         510                               Buyer   ALIAS           517         530          Buyer Entities 0.69863707
has_collective_alias   ALIAS           517         530                      Buyer Entities   ALIAS           575         579                   Buyer 0.55453944
has_collective_alias   ALIAS           517         530                      Buyer Entities   ALIAS           612         616                   Party 0.55453944
has_collective_alias   ALIAS           517         530                      Buyer Entities   ALIAS           642         648                 Parties 0.55453944
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_contract_doc_parties|
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
label                   Recall  Precision        F1   Support
dated_as                 0.962      0.962     0.962        26
has_alias                0.936      0.946     0.941        94
has_collective_alias     1.000      1.000     1.000         7
no_rel                   0.982      0.980     0.981       497
signed_by                0.961      0.961     0.961        76
Avg.                     0.968      0.970     0.969         -
Weighted-Avg.            0.973      0.973     0.973         -
``` 
