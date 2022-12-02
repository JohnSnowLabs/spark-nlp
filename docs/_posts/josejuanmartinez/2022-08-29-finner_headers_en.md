---
layout: model
title: Financial NER (Headers / Subheaders)
author: John Snow Labs
name: finner_headers
date: 2022-08-29
tags: [en, finance, ner, headers, splitting, sections, licensed]
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

This is a Named Entity Recognition model, which will help you split long financial documents into smaller sections. To do that, it detects Headers and Subheaders of different sections. You can then use the beginning and end information in the metadata to retrieve the text between those headers.

This model has been trained on 10-K filings, with the following HEADER and SUBHEADERS annotation guidelines:
- PART I, PART II, etc are HEADERS 
- Item 1, Item 2, etc are also HEADERS 
- Item 1A, 2B, etc are SUBHEADERS 
- 1., 2., 2.1, etc. are SUBHEADERS
- Other kind of short section names are also SUBHEADERS

For more information about long document splitting, see [this](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Finance/1.Tokenization_Splitting.ipynb) workshop entry.

## Predicted Entities

`HEADER`, `SUBHEADER`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINNER_HEADERS/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_headers_en_1.0.0_3.2_1661771922923.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained('finner_headers', 'en', 'finance/models')\
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
+-----------+-----------+----------+
|      token|  ner_label|confidence|
+-----------+-----------+----------+
|          2|          O|     0.576|
|          .|          O|    0.9612|
|Definitions|B-SUBHEADER|    0.9993|
|          .|          O|    0.9755|
|        For|          O|    0.9966|
|   purposes|          O|    0.9863|
|         of|          O|    0.9878|
|       this|          O|    0.9974|
|  Agreement|          O|    0.9994|
|          ,|          O|    0.9999|
|        the|          O|       1.0|
|  following|          O|       1.0|
|      terms|          O|       1.0|
|       have|          O|       1.0|
|        the|          O|       1.0|
|   meanings|          O|       1.0|
|   ascribed|          O|       1.0|
|    thereto|          O|       1.0|
|         in|          O|       1.0|
|       this|          O|       1.0|
|    Section|          O|    0.9985|
|          1|          O|    0.9999|
|          .|          O|    0.9972|
|          2|          O|    0.9686|
|          .|          O|    0.9834|
|Appointment|B-SUBHEADER|     0.767|
|         as|I-SUBHEADER|    0.9479|
|   Reseller|I-SUBHEADER|    0.8429|
|          .|          O|    0.9944|
|        2.1|B-SUBHEADER|    0.6278|
|Appointment|I-SUBHEADER|    0.6599|
|          .|          O|    0.9972|
|        The|          O|    0.9987|
|    Company|          O|    0.9889|
|     hereby|          O|    0.9914|
|      [***]|          O|    0.9996|
|          .|          O|    0.9999|
| Allscripts|          O|    0.9843|
|        may|          O|    0.9989|
|       also|          O|    0.9967|
|   disclose|          O|    0.9949|
|  Company's|          O|    0.9976|
|    pricing|          O|    0.9999|
|information|          O|    0.9999|
|   relating|          O|    0.9999|
|         to|          O|    0.9998|
|        its|          O|    0.9992|
|   Merchant|          O|    0.9671|
| Processing|          O|    0.8411|
|   Services|          O|    0.9662|
+-----------+-----------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_headers|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.3 MB|

## References

In-house annotations on 10-k filings

## Benchmarking

```bash
label            tp      fp      fn      prec         rec          f1
I-HEADER         2835    9       8       0.996835     0.9971860    0.9970107
B-SUBHEADER      963     135     131     0.877049     0.8802559    0.87864965
I-SUBHEADER      2573    219     152     0.921561     0.9442202    0.9327533
B-HEADER         425     1       1       0.997652     0.9976526    0.9976526
Macro-average    6796    364     292     0.948274     0.9548287    0.95154047
Micro-average    6796    364     292     0.949162     0.9588036    0.9539584
``` 
