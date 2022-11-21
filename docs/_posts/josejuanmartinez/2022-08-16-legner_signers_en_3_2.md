---
layout: model
title: Legal NER (Headers / Subheaders)
author: John Snow Labs
name: legner_signers
date: 2022-08-16
tags: [en, legal, ner, agreements, licensed]
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

This is a Legal NER Model, aimed to process the last page of the agreements when information can be found about:
- People Signing the document;
- Title of those people in their companies;
- Company (Party) they represent;

This model can be used all along with its Relation Extraction model to retrieve the relations between these entities.

Other models can be found to detect other parts of the document, as Headers/Subheaders, Signers, "Will-do", etc.

## Predicted Entities

`SIGNING_TITLE`, `SIGNING_PERSON`, `PARTY`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/LEGALNER_SIGNERS/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_signers_en_1.0.0_3.2_1660646474494.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = legal.NerModel.pretrained('legner_signers', 'en', 'legal/models')\
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

text = """
  VENDOR:
  VENDINGDATA CORPORATION, a Nevada corporation
  By: /s/ Steven J. Blad
  Its: Steven J. Blad CEO
  DISTRIBUTOR:
  TECHNICAL CASINO SUPPLIES LTD, an English company
  By: /s/ David K. Heap
  Its: David K. Heap Chief Executive Officer
-15-"""

res = model.transform(spark.createDataFrame([[text]]).toDF("text"))
```

</div>

## Results

```bash
+-----------+----------------+
|      token|       ner_label|
+-----------+----------------+
|     VENDOR|               O|
|          :|               O|
|VENDINGDATA|         B-PARTY|
|CORPORATION|               O|
|          ,|               O|
|          a|               O|
|     Nevada|               O|
|corporation|               O|
|         By|               O|
|          :|               O|
|        /s/|               O|
|     Steven|B-SIGNING_PERSON|
|          J|I-SIGNING_PERSON|
|          .|I-SIGNING_PERSON|
|       Blad|I-SIGNING_PERSON|
|        Its|               O|
|          :|               O|
|     Steven|B-SIGNING_PERSON|
|          J|I-SIGNING_PERSON|
|          .|I-SIGNING_PERSON|
|       Blad|I-SIGNING_PERSON|
|        CEO| B-SIGNING_TITLE|
|DISTRIBUTOR|               O|
|          :|               O|
|  TECHNICAL|         B-PARTY|
|     CASINO|         I-PARTY|
|   SUPPLIES|         I-PARTY|
|        LTD|         I-PARTY|
|          ,|               O|
|         an|               O|
|    English|               O|
|    company|               O|
|         By|               O|
|          :|               O|
|        /s/|               O|
|      David|B-SIGNING_PERSON|
|          K|I-SIGNING_PERSON|
|          .|I-SIGNING_PERSON|
|       Heap|I-SIGNING_PERSON|
|        Its|               O|
|          :|               O|
|      David|B-SIGNING_PERSON|
|          K|I-SIGNING_PERSON|
|          .|I-SIGNING_PERSON|
|       Heap|I-SIGNING_PERSON|
|      Chief| B-SIGNING_TITLE|
|  Executive| I-SIGNING_TITLE|
|    Officer| I-SIGNING_TITLE|
|          -|               O|
|         15|               O|
|          -|               O|
+-----------+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_signers|
|Type:|legal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.5 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
label               tp     fp    fn     prec           rec           f1
I-PARTY             366    26    39     0.93367344     0.9037037     0.91844416
I-SIGNING_TITLE     41     0     4      1.0            0.9111111     0.95348835
I-SIGNING_PERSON    115    10    13     0.92           0.8984375     0.9090909
B-SIGNING_PERSON    46     3     11     0.93877554     0.80701756    0.8679246
B-PARTY             122    14    28     0.89705884     0.81333333    0.85314685
B-SIGNING_TITLE     26     0     2      1.0            0.9285714     0.9629629
Macro-average	    716    53    97     0.9482513      0.8770291     0.91125065
Micro-average	    716    53    97     0.9310793      0.8806888     0.9051833
```