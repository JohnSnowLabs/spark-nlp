---
layout: model
title: Legal Relation Extraction (Parties, Alias, Dates, Document Type) (Md, Undirectional)
author: John Snow Labs
name: legre_contract_doc_parties_md
date: 2022-11-02
tags: [en, legal, licensed, re, agreements]
task: Relation Extraction
language: en
edition: Spark NLP for Legal 1.0.0
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
- Use the `legclf_introduction_clause` Text Classifier to select only these paragraphs; 
- 
This is a Legal Relation Extraction model, which can be used after the NER Model for extracting Parties, Document Types, Effective Dates and Aliases, called legner_contract_doc_parties.

As an output, you will get the relations linking the different concepts together, if such relation exists. The list of relations is:

- dated_as: A Document has an Effective Date
- has_alias: The Alias of a Party all along the document
- has_collective_alias: An Alias hold by several parties at the same time
- signed_by: Between a Party and the document they signed

This is a `md` model with Unidirectional Relations, meaning that the model retrieves in chunk1 the left side of the relation (source), and in chunk2 the right side (target).

## Predicted Entities

`dated_as`, `has_alias`, `has_collective_alias`, `signed_by`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_contract_doc_parties_md_en_1.0.0_3.0_1667404651340.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
    .setInputCols("document")\
    .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base", "en") \
    .setInputCols("document", "token")\
    .setOutputCol("embeddings")\
    .setMaxSentenceLength(512)

ner_model = legal.NerModel.pretrained('legner_contract_doc_parties', 'en', 'legal/models')\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["document","token","ner"])\
    .setOutputCol("ner_chunk")

re_model = legal.RelationExtractionDLModel().pretrained('legre_contract_doc_parties_md', 'en', 'legal/models')\
    .setPredictionThreshold(0.5)\
    .setInputCols(["ner_chunk", "document"])\
    .setOutputCol("relations")

nlpPipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter,
    re_model
    ])

empty_df = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_df)

text="""This INTELLECTUAL PROPERTY AGREEMENT (this "Agreement"), dated as of December 31, 2018 (the "Effective Date") is entered into by and between Armstrong Flooring, Inc., a Delaware corporation ("Seller") and AFI Licensing LLC, a Delaware limited liability company ("Licensing" and together with Seller, "Arizona") and AHF Holding, Inc. (formerly known as Tarzan HoldCo, Inc.), a Delaware corporation ("Buyer") and Armstrong Hardwood Flooring Company, a Tennessee corporation (the "Company" and together with Buyer the "Buyer Entities") (each of Arizona on the one hand and the Buyer Entities on the other hand, a "Party" and collectively, the "Parties")."""

data = spark.createDataFrame([[text]]).toDF("text")

result = model.transform(data)
```

</div>

## Results

```bash
| relation             | entity1 | entity1_begin | entity1_end | chunk1                          | entity2 | entity2_begin | entity2_end | chunk2                              | confidence |
|----------------------|---------|---------------|-------------|---------------------------------|---------|---------------|-------------|-------------------------------------|------------|
| dated_as             | DOC     | 6             | 36          | INTELLECTUAL PROPERTY AGREEMENT | EFFDATE | 70            | 86          | December 31, 2018                   | 0.99994016 |
| signed_by            | DOC     | 6             | 36          | INTELLECTUAL PROPERTY AGREEMENT | PARTY   | 142           | 164         | Armstrong Flooring, Inc             | 0.9995191  |
| signed_by            | DOC     | 6             | 36          | INTELLECTUAL PROPERTY AGREEMENT | ALIAS   | 193           | 198         | Seller                              | 0.9823355  |
| signed_by            | DOC     | 6             | 36          | INTELLECTUAL PROPERTY AGREEMENT | PARTY   | 206           | 222         | AFI Licensing LLC                   | 0.9989542  |
| signed_by            | DOC     | 6             | 36          | INTELLECTUAL PROPERTY AGREEMENT | ALIAS   | 264           | 272         | Licensing                           | 0.92109    |
| signed_by            | DOC     | 6             | 36          | INTELLECTUAL PROPERTY AGREEMENT | ALIAS   | 293           | 298         | Seller                              | 0.9938019  |
| signed_by            | DOC     | 6             | 36          | INTELLECTUAL PROPERTY AGREEMENT | PARTY   | 316           | 331         | AHF Holding, Inc                    | 0.9989403  |
| signed_by            | DOC     | 6             | 36          | INTELLECTUAL PROPERTY AGREEMENT | ALIAS   | 400           | 404         | Buyer                               | 0.89959186 |
| signed_by            | DOC     | 6             | 36          | INTELLECTUAL PROPERTY AGREEMENT | PARTY   | 412           | 446         | Armstrong Hardwood Flooring Company | 0.9974464  |
| signed_by            | DOC     | 6             | 36          | INTELLECTUAL PROPERTY AGREEMENT | ALIAS   | 479           | 485         | Company                             | 0.95839113 |
| signed_by            | DOC     | 6             | 36          | INTELLECTUAL PROPERTY AGREEMENT | ALIAS   | 506           | 510         | Buyer                               | 0.95839113 |
| signed_by            | DOC     | 6             | 36          | INTELLECTUAL PROPERTY AGREEMENT | ALIAS   | 517           | 530         | Buyer Entities                      | 0.95839113 |
| signed_by            | DOC     | 6             | 36          | INTELLECTUAL PROPERTY AGREEMENT | ALIAS   | 612           | 616         | Party                               | 0.95839113 |
| signed_by            | DOC     | 6             | 36          | INTELLECTUAL PROPERTY AGREEMENT | ALIAS   | 642           | 648         | Parties                             | 0.95839113 |
| dated_as             | EFFDATE | 70            | 86          | December 31, 2018               | PARTY   | 142           | 164         | Armstrong Flooring, Inc             | 0.69556713 |
| dated_as             | ALIAS   | 193           | 198         | Seller                          | EFFDATE | 70            | 86          | December 31, 2018                   | 0.7183331  |
| dated_as             | EFFDATE | 70            | 86          | December 31, 2018               | ALIAS   | 264           | 272         | Licensing                           | 0.7500907  |
| dated_as             | EFFDATE | 70            | 86          | December 31, 2018               | ALIAS   | 293           | 298         | Seller                              | 0.6601122  |
| dated_as             | EFFDATE | 70            | 86          | December 31, 2018               | PARTY   | 316           | 331         | AHF Holding, Inc                    | 0.52062315 |
| dated_as             | EFFDATE | 70            | 86          | December 31, 2018               | ALIAS   | 400           | 404         | Buyer                               | 0.7104727  |
| dated_as             | EFFDATE | 70            | 86          | December 31, 2018               | PARTY   | 412           | 446         | Armstrong Hardwood Flooring Company | 0.70473474 |
| dated_as             | ALIAS   | 479           | 485         | Company                         | EFFDATE | 70            | 86          | December 31, 2018                   | 0.98484945 |
| dated_as             | ALIAS   | 506           | 510         | Buyer                           | EFFDATE | 70            | 86          | December 31, 2018                   | 0.98484945 |
| dated_as             | ALIAS   | 517           | 530         | Buyer Entities                  | EFFDATE | 70            | 86          | December 31, 2018                   | 0.98484945 |
| dated_as             | ALIAS   | 612           | 616         | Party                           | EFFDATE | 70            | 86          | December 31, 2018                   | 0.98484945 |
| dated_as             | ALIAS   | 642           | 648         | Parties                         | EFFDATE | 70            | 86          | December 31, 2018                   | 0.98484945 |
| has_alias            | PARTY   | 142           | 164         | Armstrong Flooring, Inc         | ALIAS   | 264           | 272         | Licensing                           | 0.686296   |
| has_alias            | PARTY   | 206           | 222         | AFI Licensing LLC               | ALIAS   | 264           | 272         | Licensing                           | 0.8194909  |
| has_collective_alias | ALIAS   | 264           | 272         | Licensing                       | PARTY   | 316           | 331         | AHF Holding, Inc                    | 0.5534526  |
| has_alias            | PARTY   | 316           | 331         | AHF Holding, Inc                | ALIAS   | 479           | 485         | Company                             | 0.52909577 |
| has_alias            | PARTY   | 316           | 331         | AHF Holding, Inc                | ALIAS   | 506           | 510         | Buyer                               | 0.52909607 |
| has_alias            | PARTY   | 316           | 331         | AHF Holding, Inc                | ALIAS   | 517           | 530         | Buyer Entities                      | 0.52909607 |
| has_alias            | PARTY   | 316           | 331         | AHF Holding, Inc                | ALIAS   | 612           | 616         | Party                               | 0.52909607 |
| has_alias            | PARTY   | 316           | 331         | AHF Holding, Inc                | ALIAS   | 642           | 648         | Parties                             | 0.52909607 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_contract_doc_parties_md|
|Compatibility:|Spark NLP for Legal 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|402.3 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
label                 Recall  Precision  F1     Support 
dated_as              1.000   0.957      0.978  44      
has_alias             0.950   0.974      0.962  40      
has_collective_alias  0.667   1.000      0.800  3       
signed_by             0.957   0.989      0.972  92      
Avg.                  0.913   0.977      0.938  -       
Weighted-Avg.         0.973   0.974      0.973  -  
```
