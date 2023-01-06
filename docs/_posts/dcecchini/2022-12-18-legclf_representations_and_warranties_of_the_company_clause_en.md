---
layout: model
title: Legal Representations And Warranties Of The Company Clause Binary Classifier
author: John Snow Labs
name: legclf_representations_and_warranties_of_the_company_clause
date: 2022-12-18
tags: [en, legal, classification, licensed, clause, bert, representations, and, warranties, of, the, company, tensorflow]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Binary Classifier (True, False) for the representations-and-warranties-of-the-company clause type. To use this model, make sure you provide enough context as an input. Adding Sentence Splitters to the pipeline will make the model see only sentences, not the whole text, so it’s better to skip it, unless you want to do Binary Classification as sentence level. If you have big legal documents, and you want to look for clauses, we recommend you to split the documents using any of the techniques available in our Spark NLP for Legal Workshop Tokenization & Splitting Tutorial (link [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Legal/1.Tokenization_Splitting.ipynb)), namely:

- Paragraph splitting (by multiline);
- Splitting by headers / subheaders;
- etc.
Take into consideration the embeddings of this model allows up to 512 tokens. If you have more than that, consider splitting in smaller pieces (you can also check the same tutorial link provided above).

This model can be combined with any of the other 200+ Legal Clauses Classifiers you will find in Models Hub, getting as an output a series of True/False values for each of the legal clause model you have added.

## Predicted Entities

`representations-and-warranties-of-the-company`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_representations_and_warranties_of_the_company_clause_en_1.0.0_3.0_1671393657125.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en")\
    .setInputCols("document")\
    .setOutputCol("sentence_embeddings")

doc_classifier = legal.ClassifierDLModel.pretrained("legclf_representations_and_warranties_of_the_company_clause", "en", "legal/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")

nlpPipeline = nlp.Pipeline(stages=[
    document_assembler, 
    embeddings,
    doc_classifier])

df = spark.createDataFrame([["YOUR TEXT HERE"]]).toDF("text")

model = nlpPipeline.fit(df)

result = model.transform(df)

```

</div>

## Results

```bash

+-------+
|result|
+-------+
|[representations-and-warranties-of-the-company]|
|[other]|
|[other]|
|[representations-and-warranties-of-the-company]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_representations_and_warranties_of_the_company_clause|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.9 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents + Lawinsider categorization

## Benchmarking

```bash

                                         label    precision    recall    f1-score    support 
                                         other         0.97      1.00        0.99         39 
 representations-and-warranties-of-the-company         1.00      0.96        0.98         28 
                                      accuracy            -         -        0.99         67 
                                     macro-avg         0.99      0.98        0.98         67 
                                  weighted-avg         0.99      0.99        0.99         67 

```