---
layout: model
title: Legal Special terms and conditions of trust Clause Binary Classifier
author: John Snow Labs
name: legal_sequence_classifier_special_terms_and_conditions_of_trust_clause
date: 2022-08-04
tags: [en, legal, classification, clauses, licensed]
task: Text Classification
language: en
edition: Spark NLP for Legal 1.0.0
spark_version: 3.2
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Binary Classifier (True, False) for the `special-terms-and-conditions-of-trust` clause type. To use this model, make sure you provide enough context as an input. Adding Sentence Splitters to the pipeline will make the model see only sentences, not the whole text, so it's better to skip it, unless you want to do Binary Classification as sentence level.

If you have big legal documents, and you want to look for clauses, we recommend you to split the documents using any of the techniques available in our Spark NLP for Legal Workshop Tokenization & Splitting Tutorial (link [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Legal/1.Tokenization_Splitting.ipynb)), namely:
- Paragraph splitting (by multiline);
- Splitting by headers / subheaders;
- etc.

Take into consideration the embeddings of this model allows up to 512 tokens. If you have more than that, consider splitting in smaller pieces (you can also check the same tutorial link provided above).

This model can be combined with any of the other 200+ Legal Clauses Classifiers you will find in Models Hub, getting as an output a series of True/False values for each of the legal clause model you have added.

## Predicted Entities

`other`, `special-terms-and-conditions-of-trust`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legal_sequence_classifier_special_terms_and_conditions_of_trust_clause_en_1.0.0_3.2_1659609674442.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
     .setInputCol("clause_text") \
     .setOutputCol("document")
  
embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")

docClassifier = ClassifierDLModel().pretrained("legal_sequence_classifier_special_terms_and_conditions_of_trust_clause", "en", "legal/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")
    
nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    embeddings,
    docClassifier])
 
df = spark.createDataFrame([["YOUR TEXT HERE"]]).toDF("clause_text")
model = nlpPipeline.fit(df)
result = model.transform(df)
```

</div>

## Results

```bash
+-------+
| result|
+-------+
|[special-terms-and-conditions-of-trust]|
|[other]|
|[other]|
|[special-terms-and-conditions-of-trust]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legal_sequence_classifier_special_terms_and_conditions_of_trust_clause|
|Type:|legal|
|Compatibility:|Spark NLP for Legal 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[category]|
|Language:|en|
|Size:|23.0 MB|

## References

News scrapped from the Internet and manual in-house annotations

## Benchmarking

```bash
                                       precision    recall  f1-score   support

                                other       1.00      1.00      1.00       132
special-terms-and-conditions-of-trust       1.00      1.00      1.00        56

                             accuracy                           1.00       188
                            macro avg       1.00      1.00      1.00       188
                         weighted avg       1.00      1.00      1.00       188

```