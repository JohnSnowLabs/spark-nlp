---
layout: model
title: Cyberbullying Classifier in Turkish texts.
author: John Snow Labs
name: classifierdl_berturk_cyberbullying
date: 2021-07-21
tags: [tr, cyberbullying, classification, public, berturk, open_source]
task: Text Classification
language: tr
edition: Spark NLP 3.1.2
spark_version: 2.4
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Identifies whether a Turkish text contains cyberbullying or not.

## Predicted Entities

`Negative`, `Positive`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_TR_CYBERBULLYING/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_TR_CYBERBULLYING.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_berturk_cyberbullying_tr_3.1.2_2.4_1626884209141.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classifierdl_berturk_cyberbullying_tr_3.1.2_2.4_1626884209141.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
berturk_embeddings = BertEmbeddings.pretrained("bert_base_turkish_uncased", "tr") \
.setInputCols("document", "lemma") \
.setOutputCol("embeddings")

embeddingsSentence = SentenceEmbeddings() \
.setInputCols(["document", "embeddings"]) \
.setOutputCol("sentence_embeddings") \
.setPoolingStrategy("AVERAGE")

document_classifier = ClassifierDLModel.pretrained('classifierdl_berturk_cyberbullying', 'tr') \
.setInputCols(["document", "sentence_embeddings"]) \
.setOutputCol("class")

berturk_pipeline = Pipeline(stages=[document_assembler, tokenizer, normalizer, stopwords_cleaner, lemma, berturk_embeddings, embeddingsSentence, document_classifier])

light_pipeline = LightPipeline(berturk_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

result = light_pipeline.annotate("""Gidişin olsun, dönüşün olmasın inşallah senin..""")
result["class"]
```
```scala
...
val berturk_embeddings = BertEmbeddings.pretrained("bert_base_turkish_uncased", "tr") 
.setInputCols("document", "lemma") 
.setOutputCol("embeddings")

val embeddingsSentence = SentenceEmbeddings() 
.setInputCols(Array("document", "embeddings")) 
.setOutputCol("sentence_embeddings") 
.setPoolingStrategy("AVERAGE")

val document_classifier = ClassifierDLModel.pretrained("classifierdl_berturk_cyberbullying", "tr") 
.setInputCols(Array("document", "sentence_embeddings")) 
.setOutputCol("class")

val berturk_pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, normalizer, stopwords_cleaner, lemma, berturk_embeddings, embeddingsSentence, document_classifier))

val light_pipeline = LightPipeline(berturk_pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

val result = light_pipeline.annotate("Gidişin olsun, dönüşün olmasın inşallah senin..")
```


{:.nlu-block}
```python
import nlu
nlu.load("tr.classify.cyberbullying").predict("""Gidişin olsun, dönüşün olmasın inşallah senin..""")
```

</div>

## Results

```bash
['Negative']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_berturk_cyberbullying|
|Compatibility:|Spark NLP 3.1.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|tr|

## Data Source

Trained on a custom dataset with Turkish Bert embeddings (BERTurk).

## Benchmarking

```bash
precision    recall  f1-score   support

Negative       0.83      0.80      0.81       970
Positive       0.84      0.87      0.86      1225

accuracy                           0.84      2195
macro avg       0.84      0.83      0.84      2195
weighted avg       0.84      0.84      0.84      2195
```