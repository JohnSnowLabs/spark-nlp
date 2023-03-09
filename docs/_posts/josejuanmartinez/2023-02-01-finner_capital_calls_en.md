---
layout: model
title: NER on Capital Calls (Small)
author: John Snow Labs
name: finner_capital_calls
date: 2023-02-01
tags: [capital, calls, en, licensed]
task: Named Entity Recognition
language: en
nav_key: models
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a `small` capital call NER, trained to extract contact and financial information from Capital Call Notices. These are the entities retrieved by the model:

```
Financial information:
FUND: Name of the Fund called
ORG: Organization asking the Fund for the Capital
AMOUNT: Amount called by ORG to FUND
DUE_DATE: Due date of the call
ACCOUNT_NAME: Organization's Bank Account Name
ACCOUNT_NUMBER: Organization's Bank Account Number
ABA: Routing Number (ABA)
BANK_ADDRESS: Contact address of the Bank

Contact information:
PHONE: Contact Phone
PERSON: Contact Person
BANK_CONTACT: Person to contact in Bank
EMAIL: Contact Email

Other additional information, not directly involved in the call:
OTHER_PERSON: Other people detected (People signing the call, people to whom is addressed the Notice, etc)
OTHER_PERCENTAGE: Percentages mentiones
OTHER_DATE: Other dates mentioned, not Due Date
OTHER_AMOUNT: Other amounts mentioned
OTHER_ADDRESS: Other addresses mentiones
OTHER_ORG: Other ORG mentiones
```

## Predicted Entities

`FUND`, `ORG`, `AMOUNT`, `DUE_DATE`, `ACCOUNT_NAME`, `ACCOUNT_NUMBER`, `BANK_ADDRESS`, `PHONE`, `PERSON`, `BANK_CONTACT`, `EMAIL`, `OTHER_PERSON`, `OTHER_PERCENTAGE`, `OTHER_DATE`, `OTHER_AMOUNT`, `OTHER_ADDRESS`, `OTHER_ORG`, `ABA`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINNER_CAPITAL_CALLS){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_capital_calls_en_1.0.0_3.0_1675250939298.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_capital_calls_en_1.0.0_3.0_1675250939298.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from pyspark.sql import functions as F

documentAssembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = nlp.SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") 

tokenizer = nlp.Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")\
    .setMaxSentenceLength(512)

ner = finance.NerModel.pretrained('finner_capital_calls', 'en', 'finance/models')\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

converter = finance.NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")\

pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            ner,
                            converter
                            ])

df = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(df)

lp = nlp.LightPipeline(model)


text = """Dear Charlotte R. Davis,

We hope this message finds you well. This is to inform you that a capital call for Upfront Ventures has been initiated. The amount requested is 7000 EUR and is due on 01.01.2024.

Kindly wire transfer the funds to the following account:

Account Green Planet Solutions LLC
Account Number 1234567-1XX
Routing Number 51903761
Bank First Republic Bank

If you require any further information, please do not hesitate to reach out to us at 3055 550818 or coxeric@example.com.

Thank you for your prompt attention to this matter.

Best regards,
James Wilson"""

result = model.transform(spark.createDataFrame([[text]]).toDF("text"))

from pyspark.sql import functions as F


result.select(F.explode(F.arrays_zip(result.ner_chunk.result, result.ner_chunk.metadata)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label"),
              F.expr("cols['1']['confidence']").alias("confidence")).show(truncate=False)
                
                
```

</div>

## Results

```bash
+--------------------------+--------------+----------+
|chunk                     |ner_label     |confidence|
+--------------------------+--------------+----------+
|Charlotte R. Davis        |OTHER_PERSON  |0.971875  |
|Upfront Ventures          |FUND          |1.0       |
|7000 EUR                  |AMOUNT        |1.0       |
|01.01.2024                |DUE_DATE      |1.0       |
|Green Planet Solutions LLC|ACCOUNT_NAME  |0.999875  |
|1234567-1XX               |ACCOUNT_NUMBER|1.0       |
|51903761                  |ABA           |1.0       |
|First Republic Bank       |BANK_NAME     |0.9999333 |
|3055 550818               |PHONE         |1.0       |
|coxeric@example.com       |EMAIL         |1.0       |
|James Wilson              |OTHER_PERSON  |1.0       |
+--------------------------+--------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_capital_calls|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.3 MB|

## References

In-house capital call notices

## Benchmarking

```bash
Total test loss: 0.3542	Avg test loss: 0.0295
label	 tp	 fp	 fn	 prec	 rec	 f1
B-PERSON	 47	 0	 0	 1.0	 1.0	 1.0
I-OTHER_PERSON	 214	 0	 0	 1.0	 1.0	 1.0
I-AMOUNT	 127	 0	 0	 1.0	 1.0	 1.0
I-OTHER_PERCENTAGE	 37	 0	 0	 1.0	 1.0	 1.0
B-OTHER_DATE	 25	 0	 0	 1.0	 1.0	 1.0
I-BANK_ADDRESS	 121	 0	 0	 1.0	 1.0	 1.0
B-AMOUNT	 170	 0	 0	 1.0	 1.0	 1.0
B-OTHER_AMOUNT	 409	 0	 0	 1.0	 1.0	 1.0
I-ORG	 311	 18	 0	 0.9452888	 1.0	 0.971875
B-PHONE	 79	 0	 0	 1.0	 1.0	 1.0
I-DUE_DATE	 153	 0	 0	 1.0	 1.0	 1.0
B-FUND	 124	 0	 0	 1.0	 1.0	 1.0
B-ABA	 97	 0	 0	 1.0	 1.0	 1.0
I-ACCOUNT_NAME	 223	 0	 0	 1.0	 1.0	 1.0
I-OTHER_DATE	 25	 0	 0	 1.0	 1.0	 1.0
I-PHONE	 119	 0	 0	 1.0	 1.0	 1.0
B-BANK_ADDRESS	 39	 0	 0	 1.0	 1.0	 1.0
B-OTHER_ORG	 139	 0	 6	 1.0	 0.95862067	 0.97887325
I-OTHER_AMOUNT	 307	 0	 0	 1.0	 1.0	 1.0
I-FUND	 131	 0	 0	 1.0	 1.0	 1.0
I-BANK_NAME	 139	 0	 0	 1.0	 1.0	 1.0
B-EMAIL	 73	 0	 0	 1.0	 1.0	 1.0
I-BANK_CONTACT	 52	 0	 0	 1.0	 1.0	 1.0
B-BANK_CONTACT	 30	 0	 0	 1.0	 1.0	 1.0
B-OTHER_PERSON	 116	 0	 0	 1.0	 1.0	 1.0
B-ACCOUNT_NAME	 97	 0	 0	 1.0	 1.0	 1.0
B-DUE_DATE	 127	 0	 0	 1.0	 1.0	 1.0
B-OTHER_ADDRESS	 11	 0	 0	 1.0	 1.0	 1.0
B-ORG	 147	 6	 0	 0.9607843	 1.0	 0.98
B-BANK_NAME	 113	 0	 0	 1.0	 1.0	 1.0
B-OTHER_PERCENTAGE	 74	 0	 0	 1.0	 1.0	 1.0
I-OTHER_ADDRESS	 38	 0	 0	 1.0	 1.0	 1.0
B-ACCOUNT_NUMBER	 97	 0	 0	 1.0	 1.0	 1.0
I-PERSON	 109	 0	 0	 1.0	 1.0	 1.0
I-OTHER_ORG	 283	 0	 18	 1.0	 0.9401993	 0.969178
I-ACCOUNT_NUMBER	 32	 0	 0	 1.0	 1.0	 1.0
Macro-average 4435  24 24 0.997391 0.9971894 0.9972902
Micro-average 4435  24 24  0.99461764 0.99461764 0.99461764
```