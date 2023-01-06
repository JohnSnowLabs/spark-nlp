---
layout: model
title: Legal Retention Of Sub Adviser Clause Binary Classifier
author: John Snow Labs
name: legclf_retention_of_sub_adviser_clause
date: 2022-12-05
tags: [en, legal, retention_of_sub_adviser, classification, clauses, licensed]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Binary Classifier (True, False) for the `retention-of-sub-adviser` clause type. To use this model, make sure you provide enough context as an input. Adding Sentence Splitters to the pipeline will make the model see only sentences, not the whole text, so it's better to skip it, unless you want to do Binary Classification as sentence level.

        If you have big legal documents, and you want to look for clauses, we recommend you to split the documents using any of the techniques available in our Legal NLP Workshop Tokenization & Splitting Tutorial (link [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Legal/1.Tokenization_Splitting.ipynb)), namely:
        - Paragraph splitting (by multiline);
        - Splitting by headers / subheaders;
        - etc.

        Take into consideration the embeddings of this model allows up to 512 tokens. If you have more than that, consider splitting in smaller pieces (you can also check the same tutorial link provided above).

        This model can be combined with any of the other 200+ Legal Clauses Classifiers you will find in Models Hub, getting as an output a series of True/False values for each of the legal clause model you have added.

## Predicted Entities

`retention-of-sub-adviser`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_retention_of_sub_adviser_clause_en_1.0.0_3.0_1670246672472.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
 documentAssembler = nlp.DocumentAssembler() \
         .setInputCol("clause_text") \
         .setOutputCol("document")

embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings") 

docClassifier = legal.ClassifierDLModel.pretrained("legclf_retention_of_sub_adviser_clause", "en", "legal/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")

nlpPipeline = nlp.Pipeline(stages=[
    documentAssembler, 
    embeddings,
    docClassifier])

df = spark.createDataFrame([["YOUR TEXT HERE"]]).toDF("clause_text")
model = nlpPipeline.fit(df)
result = model.transform(df)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_retention_of_sub_adviser_clause|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.7 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house

## Benchmarking

```bash
                   label  precision    recall  f1-score   support
                   other       0.97      1.00      0.99        34
retention-of-sub-adviser       1.00      0.95      0.97        19
                accuracy          -         -      0.98        53
               macro-avg       0.99      0.97      0.98        53
            weighted-avg       0.98      0.98      0.98        53
```