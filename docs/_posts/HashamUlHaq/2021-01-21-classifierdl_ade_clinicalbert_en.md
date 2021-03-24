---
layout: model
title: Classifier for Adverse Drug Events using Clinical Bert
author: John Snow Labs
name: classifierdl_ade_clinicalbert
date: 2021-01-21
task: Text Classification
language: en
edition: Spark NLP for Healthcare 2.7.1
tags: [en, licensed, classifier, clinical]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Classify text/sentence in two categories:

 `True` : The sentence is talking about a possible ADE

 `False` : The sentences doesn’t have any information about an ADE.

## Predicted Entities

`True`, `False`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PP_ADE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/classifierdl_ade_clinicalbert_en_2.7.1_2.4_1611244439637.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

tokenizer = Tokenizer().setInputCols(['document']).setOutputCol('token')

embeddings = BertEmbeddings.pretrained('biobert_clinical_base_cased')\
    .setInputCols(["document", 'token'])\
    .setOutputCol("word_embeddings")

sentence_embeddings = SentenceEmbeddings() \
      .setInputCols(["document", "word_embeddings"]) \
      .setOutputCol("sentence_embeddings") \
      .setPoolingStrategy("AVERAGE")

classifier = ClassifierDLModel.pretrained('classifierdl_ade_clinicalbert', 'en', 'clinical/models')\
    .setInputCols(['document', 'token', 'sentence_embeddings']).setOutputCol('class')

nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, embeddings, sentence_embeddings, classifier])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate(["I feel a bit drowsy & have a little blurred vision after taking an insulin", "I feel great after taking tylenol"])
```

</div>

## Results

```bash
|   | text                                                                       | label |
|--:|:---------------------------------------------------------------------------|:------|
| 0 | I feel a bit drowsy & have a little blurred vision after taking an insulin | True  |
| 1 | I feel great after taking tylenol                                          | False |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_ade_clinicalbert|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Dependencies:|biobert_clinical_base_cased|

## Data Source

Trained on a custom dataset comprising of CADEC, DRUG-AE and Twimed.

## Benchmarking

```bash
              precision    recall  f1-score   support

       False       0.95      0.92      0.93      6923
        True       0.64      0.78      0.70      1359

   micro avg       0.89      0.89      0.89      8282
   macro avg       0.80      0.85      0.82      8282
weighted avg       0.90      0.89      0.90      8282
```