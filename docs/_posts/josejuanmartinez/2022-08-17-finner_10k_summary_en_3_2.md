---
layout: model
title: Financial 10K Filings NER
author: John Snow Labs
name: finner_10k_summary
date: 2022-08-17
tags: [en, finance, ner, annual, reports, 10k, filings, licensed]
task: Named Entity Recognition
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

IMPORTANT: Don't run this model on the whole financial report. Instead:
- Split by paragraphs;
- Use the `finclf_form_10k_summary_item` Text Classifier to select only these paragraphs;

This Financial NER Model is aimed to process the first summary page of 10K filings and extract the information about the Company submitting the filing, trading data, address / phones, CFN, IRS, etc.

## Predicted Entities

`ADDRESS`, `CFN`, `FISCAL_YEAR`, `IRS`, `ORG`, `PHONE`, `STATE`, `STOCK_EXCHANGE`, `TICKER`, `TITLE_CLASS`, `TITLE_CLASS_VALUE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINNER_SEC10K_FIRSTPAGE/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10k_summary_en_1.0.0_3.2_1660732829888.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_10k_summary_en_1.0.0_3.2_1660732829888.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = nlp.SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") \
    .setCustomBounds(["\n\n"])

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_finbert_pretrain_yiyanghkust","en")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")\
    .setCaseSensitive(True)\
    .setMaxSentenceLength(512)

ner_model = finance.NerModel.pretrained("finner_10k_summary","en","finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")\

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter   
    ])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame([["""ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES AND EXCHANGE ACT OF 1934
For the annual period ended January 31, 2021
or
TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
For the transition period from________to_______
Commission File Number: 001-38856
PAGERDUTY, INC.
(Exact name of registrant as specified in its charter)
Delaware
27-2793871
(State or other jurisdiction of
incorporation or organization)
(I.R.S. Employer
Identification Number)
600 Townsend St., Suite 200, San Francisco, CA 94103
(844) 800-3889
(Address, including zip code, and telephone number, including area code, of registrant’s principal executive offices)
Securities registered pursuant to Section 12(b) of the Act:
Title of each class
Trading symbol(s)
Name of each exchange on which registered
Common Stock, $0.000005 par value,
PD
New York Stock Exchange"""]]).toDF("text")

result = model.transform(data)

result.select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata')).alias("cols")) \
               .select(F.expr("cols['0']").alias("ticker"),
                       F.expr("cols['1']['entity']").alias("label")).show(50, truncate = False)
```

</div>

## Results

```bash
+----------------------------------------------+-----------------+
|ticker                                        |label            |
+----------------------------------------------+-----------------+
|January 31, 2021                              |FISCAL_YEAR      |
|001-38856                                     |CFN              |
|PAGERDUTY, INC                                |ORG              |
|Delaware                                      |STATE            |
|27-2793871                                    |IRS              |
|600 Townsend St., Suite 200, San Francisco, CA|ADDRESS          |
|(844) 800-3889                                |PHONE            |
|Common Stock                                  |TITLE_CLASS      |
|$0.000005                                     |TITLE_CLASS_VALUE|
|PD                                            |TICKER           |
|New York Stock Exchange                       |STOCK_EXCHANGE   |
+----------------------------------------------+-----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10k_summary|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|1.1 MB|

## References

Manual annotations on 10-K Filings

## Benchmarking

```bash
              label  precision    recall  f1-score   support
      B-TITLE_CLASS       1.00      1.00      1.00        15
      I-TITLE_CLASS       1.00      1.00      1.00        21
              B-ORG       0.84      0.66      0.74        62
              I-ORG       0.88      0.76      0.82        93
   B-STOCK_EXCHANGE       0.86      0.86      0.86        14
   I-STOCK_EXCHANGE       0.98      0.98      0.98        50
            B-PHONE       0.95      0.87      0.91        23
            I-PHONE       0.95      1.00      0.98        60
            B-STATE       0.89      0.85      0.87        20
              B-IRS       1.00      0.88      0.93        16
          B-ADDRESS       0.94      0.83      0.88        18
          I-ADDRESS       0.92      0.97      0.94       144
           B-TICKER       0.86      0.92      0.89        13
      B-FISCAL_YEAR       0.96      0.88      0.92        50
      I-FISCAL_YEAR       0.93      0.92      0.92       125
B-TITLE_CLASS_VALUE       1.00      0.93      0.97        15
              B-CFN       0.92      1.00      0.96        12

          micro avg       0.93      0.89      0.91       751
          macro avg       0.84      0.81      0.82       751
       weighted avg       0.92      0.89      0.91       751
```
