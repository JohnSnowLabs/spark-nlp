---
layout: model
title: Legal Roles NER
author: John Snow Labs
name: legner_roles
date: 2023-01-02
tags: [role, en, licensed]
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

This NER models extracts legal roles in an agreement, such as `Borrower`, `Supplier`, `Agent`, `Attorney`, `Pursuant`, etc.

## Predicted Entities

`ROLE`, `O`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_roles_en_1.0.0_3.0_1672673551040.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencizer = nlp.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner = legal.NerModel.pretrained('legner_roles', 'en', 'legal/models')\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")\

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = nlp.Pipeline(stages=[documenter, sentencizer, tokenizer, embeddings, ner, ner_converter])

empty = spark.createDataFrame([[example]]).toDF("text")

tr_results = model.transform(spark.createDataFrame([[example]]).toDF('text'))

```

</div>

## Results

```bash
+-------+---------+---------+
|sent_id|chunk    |ner_label|
+-------+---------+---------+
|1      |Lender   |ROLE     |
|1      |Lender's |ROLE     |
|1      |principal|ROLE     |
|1      |Lender   |ROLE     |
|2      |pursuant |ROLE     |
|3      |Lenders  |ROLE     |
|3      |Lenders  |ROLE     |
|3      |Lenders  |ROLE     |
|4      |Lenders  |ROLE     |
|7      |Agent    |ROLE     |
|14     |Lenders  |ROLE     |
|14     |Borrowers|ROLE     |
|14     |Lender   |ROLE     |
|14     |Agent    |ROLE     |
|14     |Lender   |ROLE     |
|15     |Agent    |ROLE     |
|15     |Lender   |ROLE     |
|15     |pursuant |ROLE     |
|15     |Agent    |ROLE     |
|15     |Borrowers|ROLE     |
+-------+---------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_roles|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.2 MB|

## References

CUAD dataset and synthetic data

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
B-ROLE	 19095	 16	 77	 0.9991628	 0.9959837	 0.9975707
I-ROLE	 162	 1	 0	 0.993865	 1.0	 0.9969231
Macro-average	19257 17 77 0.9965139 0.99799186 0.9972523
Micro-average	19257 17 77 0.999118 0.9960174 0.9975653

```