---
layout: model
title: Legal Indemnification NER (Bert, sm)
author: John Snow Labs
name: legner_bert_indemnifications
date: 2022-09-27
tags: [indemnifications, en, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_indemnification_clause` Text Classifier to select only these paragraphs; 

This is a Legal Named Entity Recognition Model to identify the Subject (who), Action (web), Object(the indemnification) and Indirect Object (to whom) from Indemnification clauses.

There is a lighter (non-transformer based) model available in Models Hub as `legner_indemnifications_md`.

## Predicted Entities

`INDEMNIFICATION`, `INDEMNIFICATION_SUBJECT`, `INDEMNIFICATION_ACTION`, `INDEMNIFICATION_INDIRECT_OBJECT`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/legal/LEGALRE_INDEMNIFICATION/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_bert_indemnifications_en_1.0.0_3.0_1664273651991.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_bert_indemnifications_en_1.0.0_3.0_1664273651991.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

sentencizer = nlp.SentenceDetectorDLModel\
        .pretrained("sentence_detector_dl", "en") \
        .setInputCols(["document"])\
        .setOutputCol("sentence")
                      
tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

tokenClassifier = legal.BertForTokenClassification.pretrained("legner_bert_indemnifications", "en", "legal/models")\
  .setInputCols("token", "sentence")\
  .setOutputCol("label")\
  .setCaseSensitive(True)

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence","token","label"])\
    .setOutputCol("ner_chunk")
    
nlpPipeline = nlp.Pipeline(stages=[
        documentAssembler,
        sentencizer,
        tokenizer,
        tokenClassifier,
        ner_converter
        ])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text='''The Company shall protect and indemnify the Supplier against any damages, losses or costs whatsoever'''

data = spark.createDataFrame([[text]]).toDF("text")
model = nlpPipeline.fit(data)
lmodel = LightPipeline(model)
res = lmodel.annotate(text)
```

</div>

## Results

```bash
+----------+---------------------------------+
|     token|                        ner_label|
+----------+---------------------------------+
|       The|                                O|
|   Company|                                O|
|     shall|         B-INDEMNIFICATION_ACTION|
|   protect|         I-INDEMNIFICATION_ACTION|
|       and|                                O|
| indemnify|         B-INDEMNIFICATION_ACTION|
|       the|                                O|
|  Supplier|B-INDEMNIFICATION_INDIRECT_OBJECT|
|   against|                                O|
|       any|                                O|
|   damages|                B-INDEMNIFICATION|
|         ,|                                O|
|    losses|                B-INDEMNIFICATION|
|        or|                                O|
|     costs|                B-INDEMNIFICATION|
|whatsoever|                                O|
+----------+---------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_bert_indemnifications|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|412.2 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

In-house annotated examples from CUAD legal dataset

## Benchmarking

```bash
                            label  precision    recall  f1-score   support
                B-INDEMNIFICATION       0.91      0.89      0.90        36
         B-INDEMNIFICATION_ACTION       0.92      0.71      0.80        17
B-INDEMNIFICATION_INDIRECT_OBJECT       0.88      0.88      0.88        40
        B-INDEMNIFICATION_SUBJECT       0.71      0.56      0.63         9
                I-INDEMNIFICATION       0.88      0.78      0.82         9
         I-INDEMNIFICATION_ACTION       0.81      0.87      0.84        15
I-INDEMNIFICATION_INDIRECT_OBJECT       1.00      0.53      0.69        17
                                O       0.97      0.91      0.94       510
                         accuracy        -          -       0.88       654
                        macro-avg       0.71      0.61      0.81       654
                     weighted-avg       0.95      0.88      0.91       654
```