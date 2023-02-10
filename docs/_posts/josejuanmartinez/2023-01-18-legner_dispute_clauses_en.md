---
layout: model
title: Dispute Clauses NER
author: John Snow Labs
name: legner_dispute_clauses
date: 2023-01-18
tags: [en, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Legal NER model which helps to retrieve Courts/Arbitrations, Rules and Resolution Means from legal agreements.

## Predicted Entities

`RESOLUT_MEANS`, `RULES_NAME`, `COURT_NAME`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_dispute_clauses_en_1.0.0_3.0_1674054944954.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = legal.NerModel().pretrained("legner_dispute_clauses","en","legal/models")\
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

text = ["""The contract includes a dispute clause that requires the parties to follow the rules of judicial arbitration set forth by the United Nations Commission on International Trade Law (UNCITRAL) Rules of Arbitration and the jurisdiction of the International Chamber of Commerce court in the event of a dispute."""]

res = model.transform(spark.createDataFrame([text]).toDF("text"))
```

</div>

## Results

```bash
+-------------+---------------+
|        token|      ner_label|
+-------------+---------------+
|          The|              O|
|     contract|              O|
|     includes|              O|
|            a|              O|
|      dispute|              O|
|       clause|              O|
|         that|              O|
|     requires|              O|
|          the|              O|
|      parties|              O|
|           to|              O|
|       follow|              O|
|          the|              O|
|        rules|              O|
|           of|              O|
|     judicial|B-RESOLUT_MEANS|
|  arbitration|I-RESOLUT_MEANS|
|          set|              O|
|        forth|              O|
|           by|              O|
|          the|              O|
|       United|   B-RULES_NAME|
|      Nations|   I-RULES_NAME|
|   Commission|   I-RULES_NAME|
|           on|   I-RULES_NAME|
|International|   I-RULES_NAME|
|        Trade|   I-RULES_NAME|
|          Law|   I-RULES_NAME|
|            (|   I-RULES_NAME|
|     UNCITRAL|   I-RULES_NAME|
|            )|   I-RULES_NAME|
|        Rules|   I-RULES_NAME|
|           of|   I-RULES_NAME|
|  Arbitration|   I-RULES_NAME|
|          and|              O|
|          the|              O|
| jurisdiction|              O|
|           of|              O|
|          the|              O|
|International|   B-COURT_NAME|
|      Chamber|   I-COURT_NAME|
|           of|   I-COURT_NAME|
|     Commerce|   I-COURT_NAME|
|        court|              O|
|           in|              O|
|          the|              O|
|        event|              O|
|           of|              O|
|            a|              O|
|      dispute|              O|
|            .|              O|
+-------------+---------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_dispute_clauses|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.2 MB|

## References

In-house annotations of the CUAD dataset

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
B-RESOLUT_MEANS	 14	 4	 6	 0.7777778	 0.7	 0.73684216
B-RULES_NAME	 15	 0	 5	 1.0	 0.75	 0.85714287
I-RESOLUT_MEANS	 12	 0	 3	 1.0	 0.8	 0.88888896
B-COURT_NAME	 26	 6	 6	 0.8125	 0.8125	 0.8125
I-RULES_NAME	 101	 7	 19	 0.9351852	 0.84166664	 0.8859649
I-COURT_NAME	 166	 23	 24	 0.87830687	 0.8736842	 0.87598944
Macro-average	 334 40 63 0.9006283 0.7963085 0.8452619
Micro-average	 334 40 63 0.8930481 0.84130985 0.8664072
```