---
layout: model
title: Legal NER (Headers / Subheaders)
author: John Snow Labs
name: legner_headers
date: 2022-08-12
tags: [en, legal, ner, agreements, splitting, licensed]
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

This is a Legal NER Model, aimed to carry out Section Splitting by using the Headers and Subheaders entities, detected in the document.

Other models can be found to detect other parts of the document, as Headers/Subheaders, Signers, "Will-do", etc.

## Predicted Entities

`HEADER`, `SUBHEADER`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/LEGALNER_HEADERS/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_headers_en_1.0.0_3.2_1660298515978.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_headers_en_1.0.0_3.2_1660298515978.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = nlp.SentenceDetector()\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained('legner_headers', 'en', 'legal/models')\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = ["""
2. Definitions. For purposes of this Agreement, the following terms have the meanings ascribed thereto in this Section 1. 2. Appointment as Reseller.

2.1 Appointment. The Company hereby [***]. Allscripts may also disclose Company's pricing information relating to its Merchant Processing Services and facilitate procurement of Merchant Processing Services on behalf of Sublicensed Customers, including, without limitation by references to such pricing information and Merchant Processing Services in Customer Agreements. 6

2.2 Customer Agreements."""]

res = model.transform(spark.createDataFrame([text]).toDF("text"))
```

</div>

## Results

```bash
+-----------+-----------+
|      token|  ner_label|
+-----------+-----------+
|          2|   B-HEADER|
|          .|   I-HEADER|
|Definitions|   I-HEADER|
|          .|          O|
|        For|          O|
|   purposes|          O|
|         of|          O|
|       this|          O|
|  Agreement|          O|
|          ,|          O|
|        the|          O|
|  following|          O|
|      terms|          O|
|       have|          O|
|        the|          O|
|   meanings|          O|
|   ascribed|          O|
|    thereto|          O|
|         in|          O|
|       this|          O|
|    Section|          O|
|          1|B-SUBHEADER|
|          .|I-SUBHEADER|
|          2|I-SUBHEADER|
|          .|I-SUBHEADER|
|Appointment|   I-HEADER|
|         as|   I-HEADER|
|   Reseller|   I-HEADER|
|          .|          O|
|        2.1|B-SUBHEADER|
|Appointment|I-SUBHEADER|
|          .|          O|
|        The|          O|
|    Company|          O|
|     hereby|          O|
|      [***]|          O|
|          .|          O|
| Allscripts|          O|
|        may|          O|
|       also|          O|
|   disclose|          O|
|  Company's|          O|
|    pricing|          O|
|information|          O|
|   relating|          O|
|         to|          O|
|        its|          O|
|   Merchant|          O|
| Processing|          O|
|   Services|          O|
|        and|          O|
| facilitate|          O|
|procurement|          O|
|         of|          O|
|   Merchant|          O|
| Processing|          O|
|   Services|          O|
|         on|          O|
|     behalf|          O|
|         of|          O|
|Sublicensed|          O|
|  Customers|          O|
|          ,|          O|
|  including|          O|
|          ,|          O|
|    without|          O|
| limitation|          O|
|         by|          O|
| references|          O|
|         to|          O|
|       such|          O|
|    pricing|          O|
|information|          O|
|        and|          O|
|   Merchant|          O|
| Processing|          O|
|   Services|          O|
|         in|          O|
|   Customer|          O|
| Agreements|          O|
|          .|          O|
|          6|          O|
|        2.2|B-SUBHEADER|
|   Customer|I-SUBHEADER|
| Agreements|I-SUBHEADER|
|          .|          O|
+-----------+-----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_headers|
|Type:|legal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.3 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
label           tp      fp    fn    prec         rec           f1
I-HEADER        1486    40    25    0.97378767   0.98345464    0.9785973
B-SUBHEADER     744     16    14    0.97894734   0.98153037    0.9802372
I-SUBHEADER     2382    53    34    0.9782341    0.98592716    0.98206556
B-HEADER        415     4     12    0.9904535    0.97189695    0.9810875
Macro-average   5027    113   85    0.9803556    0.9807023     0.9805289
Micro-average   5027    113   85    0.97801554   0.98337245    0.98068666
```


