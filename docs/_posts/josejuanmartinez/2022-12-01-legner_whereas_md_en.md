---
layout: model
title: Legal NER - Whereas Clauses (Md)
author: John Snow Labs
name: legner_whereas_md
date: 2022-12-01
tags: [whereas, en, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_cuad_whereas_clause` Text Classifier to select only these paragraphs; 

This is a Legal NER Model, able to process WHEREAS clauses, to detect the SUBJECT (Who?), the ACTION, the OBJECT (what?) and, in some cases, the INDIRECT OBJECT (to whom?) of the clause.

This is a `md` (medium version) of the classifier, trained with more data and being more resistent to false positives outside the specific section, which may help to run it at whole document level (although not recommended).

## Predicted Entities

`WHEREAS_SUBJECT`, `WHEREAS_OBJECT`, `WHEREAS_ACTION`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/LEGALNER_WHEREAS/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_whereas_md_en_1.0.0_3.0_1669892674388.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained('legner_whereas_md', 'en', 'legal/models')\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = nlp.Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = ["""WHEREAS, Seller and Buyer have entered into that certain Stock Purchase Agreement, dated November 14, 2018 (the "Stock Purchase Agreement"); WHEREAS, pursuant to the Stock Purchase Agreement, Seller has agreed to sell and transfer, and Buyer has agreed to purchase and acquire, all of Seller's right, title and interest in and to Armstrong Wood Products, Inc., a Delaware corporation ("AWP") and its Subsidiaries, the Company and HomerWood Hardwood Flooring Company, a Delaware corporation ("HHFC," and together with the Company, the "Company Subsidiaries" and together with AWP, the "Company Entities" and each a "Company Entity") by way of a purchase by Buyer and sale by Seller of the Shares, all upon the terms and condition set forth therein;"""]

res = model.transform(spark.createDataFrame([text]).toDF("text"))
```

</div>

## Results

```bash
+------------+-----------------+
|       token|        ner_label|
+------------+-----------------+
|     WHEREAS|                O|
|           ,|                O|
|      Seller|B-WHEREAS_SUBJECT|
|         and|                O|
|       Buyer|B-WHEREAS_SUBJECT|
|        have| B-WHEREAS_ACTION|
|     entered| I-WHEREAS_ACTION|
|        into| I-WHEREAS_ACTION|
|        that| B-WHEREAS_OBJECT|
|     certain| I-WHEREAS_OBJECT|
|       Stock| I-WHEREAS_OBJECT|
|    Purchase| I-WHEREAS_OBJECT|
|   Agreement| I-WHEREAS_OBJECT|
|           ,|                O|
|       dated|                O|
|    November|                O|
|          14|                O|
|           ,|                O|
|        2018|                O|
|           (|                O|
|         the|                O|
|           "|                O|
|       Stock|                O|
|    Purchase|                O|
|   Agreement|                O|
|         ");|                O|
|     WHEREAS|                O|
|           ,|                O|
|    pursuant|                O|
|          to|                O|
|         the|                O|
|       Stock|                O|
|    Purchase|                O|
|   Agreement|                O|
|           ,|                O|
|      Seller|B-WHEREAS_SUBJECT|
|         has| B-WHEREAS_ACTION|
|      agreed| I-WHEREAS_ACTION|
|          to| I-WHEREAS_ACTION|
|        sell| I-WHEREAS_ACTION|
|         and|                O|
|    transfer|                O|
|           ,|                O|
|         and|                O|
|       Buyer|B-WHEREAS_SUBJECT|
|         has| B-WHEREAS_ACTION|
|      agreed| I-WHEREAS_ACTION|
|          to| I-WHEREAS_ACTION|
|    purchase| I-WHEREAS_ACTION|
|         and|                O|
|     acquire|                O|
|           ,|                O|
|         all|                O|
|          of|                O|
|    Seller's|                O|
|       right|                O|
|           ,|                O|
|       title|                O|
|         and|                O|
|    interest|                O|
|          in|                O|
|         and|                O|
|          to|                O|
|   Armstrong|                O|
|        Wood|                O|
|    Products|                O|
|           ,|                O|
|         Inc|                O|
|          .,|                O|
|           a|                O|
|    Delaware|                O|
| corporation|                O|
|          ("|                O|
|         AWP|                O|
|          ")|                O|
|         and|                O|
|         its|                O|
|Subsidiaries|                O|
|           ,|                O|
|         the|                O|
|     Company|                O|
|         and|                O|
|   HomerWood|                O|
|    Hardwood|                O|
|    Flooring|                O|
|     Company|                O|
|           ,|                O|
|           a|                O|
|    Delaware|                O|
| corporation|                O|
|          ("|                O|
|        HHFC|                O|
|          ,"|                O|
+------------+-----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_whereas_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.1 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
B-WHEREAS_SUBJECT	 95	 12	 5	 0.88785046	 0.95	 0.9178744
I-WHEREAS_ACTION	 112	 36	 13	 0.7567568	 0.896	 0.82051283
I-WHEREAS_SUBJECT	 31	 6	 6	 0.8378378	 0.8378378	 0.8378378
B-WHEREAS_OBJECT	 59	 33	 30	 0.6413044	 0.66292137	 0.6519337
B-WHEREAS_ACTION	 87	 12	 3	 0.8787879	 0.96666664	 0.9206349
I-WHEREAS_OBJECT	 221	 108	 65	 0.67173254	 0.77272725	 0.71869916
Macro-average	 605 207 122 0.77904505 0.8476922 0.81192017
Micro-average	 605 207 122 0.7450739 0.83218706 0.78622484
```
