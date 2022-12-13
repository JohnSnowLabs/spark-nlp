---
layout: model
title: Legal Confidentiality NER
author: John Snow Labs
name: legner_confidentiality
date: 2022-10-17
tags: [legal, en, ner, licensed, confidentiality]
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
- Use the `legclf_cuad_confidentiality_clause` Text Classifier to select only these paragraphs; 

This is a Legal Named Entity Recognition Model to identify the Subject (who), Action (web), Object(the indemnification) and Indirect Object (to whom) from Confidentiality clauses.

## Predicted Entities

`CONFIDENTIALITY`, `CONFIDENTIALITY_ACTION`, `CONFIDENTIALITY_INDIRECT_OBJECT`, `CONFIDENTIALITY_SUBJECT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_confidentiality_en_1.0.0_3.0_1666013443039.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_confidentiality_en_1.0.0_3.0_1666013443039.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = legal.NerModel.pretrained('legner_confidentiality', 'en', 'legal/models')\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[documentAssembler,sentenceDetector,tokenizer,embeddings,ner_model,ner_converter])

data = spark.createDataFrame([["Each party will promptly return to the other upon request any Confidential Information of the other party then in its possession or under its control."]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)
```

</div>

## Results

```bash
+------------------------+-------------------------------+
|chunk                   |entity                         |
+------------------------+-------------------------------+
|Each party              |CONFIDENTIALITY_SUBJECT        |
|will promptly return    |CONFIDENTIALITY_ACTION         |
|other                   |CONFIDENTIALITY_INDIRECT_OBJECT|
|Confidential Information|CONFIDENTIALITY                |
+------------------------+-------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_confidentiality|
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
                                   precision    recall  f1-score   support
                B-CONFIDENTIALITY     0.9077    0.9219    0.9147        64
         B-CONFIDENTIALITY_ACTION     1.0000    1.0000    1.0000        53
B-CONFIDENTIALITY_INDIRECT_OBJECT     0.9419    0.9529    0.9474        85
        B-CONFIDENTIALITY_SUBJECT     0.9697    1.0000    0.9846        32
                I-CONFIDENTIALITY     0.9302    0.9091    0.9195        88
         I-CONFIDENTIALITY_ACTION     1.0000    0.9825    0.9912        57
I-CONFIDENTIALITY_INDIRECT_OBJECT     0.9744    0.8444    0.9048        45
        I-CONFIDENTIALITY_SUBJECT     1.0000    1.0000    1.0000        25
                                O     0.9913    0.9950    0.9932      1604
                         accuracy                         0.9839      2053
                        macro avg     0.9683    0.9562    0.9617      2053
                     weighted avg     0.9839    0.9839    0.9838      2053
```
