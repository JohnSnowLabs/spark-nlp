---
layout: model
title: Legal NER - Warranties (sm)
author: John Snow Labs
name: legner_warranty
date: 2022-10-17
tags: [legal, en, ner, licensed, warranty]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_warranty_clause` Text Classifier to select only these paragraphs; 

This is a Legal Named Entity Recognition Model to identify the Subject (who), Action (what), Object(the indemnification) and Indirect Object (to whom) from Warranty clauses.

## Predicted Entities

`WARRANTY`, `WARRANTY_ACTION`, `WARRANTY_SUBJECT`, `WARRANTY_INDIRECT_OBJECT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_warranty_en_1.0.0_3.0_1666013884679.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_warranty_en_1.0.0_3.0_1666013884679.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = legal.NerModel.pretrained('legner_warranty', 'en', 'legal/models')\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = nlp.Pipeline(stages=[documentAssembler,sentenceDetector,tokenizer,embeddings,ner_model,ner_converter])

data = spark.createDataFrame([["""8 . Representations and Warranties SONY hereby makes the following representations and warranties to PURCHASER , each of which shall be true and correct as of the date hereof and as of the Closing Date , and shall be unaffected by any investigation heretofore or hereafter made : 8.1 Power and Authority SONY has the right and power to enter into this IP Agreement and to transfer the Transferred Patents and to grant the license set forth in Section 3.1 ."""]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)
```

</div>

## Results

```bash
+--------------------------------------------------------------------------+------------------------+
|chunk                                                                     |entity                  |
+--------------------------------------------------------------------------+------------------------+
|SONY                                                                      |WARRANTY_SUBJECT        |
|makes the following representations and warranties                        |WARRANTY_ACTION         |
|PURCHASER                                                                 |WARRANTY_INDIRECT_OBJECT|
|shall be true and correct as of the date hereof and as of the Closing Date|WARRANTY                |
|shall be unaffected by any investigation                                  |WARRANTY                |
|SONY                                                                      |WARRANTY_SUBJECT        |
|has the right and power to enter into this IP Agreement                   |WARRANTY                |
+--------------------------------------------------------------------------+------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_warranty|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.3 MB|

## References

In-house annotated examples from CUAD legal dataset

## Benchmarking

```bash
                    label     precision    recall  f1-score   support
                B-WARRANTY     0.8993    0.9178    0.9085       146
         B-WARRANTY_ACTION     1.0000    0.9318    0.9647        44
B-WARRANTY_INDIRECT_OBJECT     1.0000    0.9474    0.9730        19
        B-WARRANTY_SUBJECT     0.8554    0.9726    0.9103        73
                I-WARRANTY     0.9695    0.9618    0.9656      1885
         I-WARRANTY_ACTION     0.9515    0.9800    0.9655       100
I-WARRANTY_INDIRECT_OBJECT     0.8333    0.8333    0.8333         6
        I-WARRANTY_SUBJECT     1.0000    0.9444    0.9714        36
                         O     0.9758    0.9772    0.9765      3381
                  accuracy        -        -       0.9698      5690
                 macro avg     0.9428    0.9407    0.9410      5690
              weighted avg     0.9700    0.9698    0.9698      5690
```
