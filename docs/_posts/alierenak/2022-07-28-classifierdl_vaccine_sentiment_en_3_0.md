---
layout: model
title: Vaccine Sentiment Classifier (PHS-BERT)
author: John Snow Labs
name: classifierdl_vaccine_sentiment
date: 2022-07-28
tags: [public_health, en, licensed, vaccine_sentiment, classification]
task: Sentiment Analysis
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: ClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a [PHS-BERT](https://arxiv.org/abs/2204.04521) based sentimental analysis model that can extract information from COVID-19 Vaccine-related tweets. The model predicts whether a tweet contains positive, negative, or neutral sentiments about COVID-19 Vaccines.

## Predicted Entities

`negative`, `positive`, `neutral`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/classifierdl_vaccine_sentiment_en_4.0.0_3.0_1658998378316.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/classifierdl_vaccine_sentiment_en_4.0.0_3.0_1658998378316.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

bert_embeddings = BertEmbeddings.pretrained("bert_embeddings_phs_bert", "en", "public/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")\

embeddingsSentence = SentenceEmbeddings() \
    .setInputCols(["sentence", "embeddings"]) \
    .setOutputCol("sentence_embeddings") \
    .setPoolingStrategy("AVERAGE")

classifierdl = ClassifierDLModel.pretrained('classifierdl_vaccine_sentiment', "en", "clinical/models")\
    .setInputCols(['sentence', 'token', 'sentence_embeddings'])\
    .setOutputCol('class')

pipeline = Pipeline(
    stages = [
        document_assembler,
        tokenizer,
        bert_embeddings,
        embeddingsSentence,
        classifierdl
    ])

text_list = ['A little bright light for an otherwise dark week. Thanks researchers, and frontline workers. Onwards.', 
             'People with a history of severe allergic reaction to any component of the vaccine should not take.', 
             '43 million doses of vaccines administrated worldwide...Production capacity of CHINA to reach 4 b']

data = spark.createDataFrame(text_list, StringType()).toDF("text")
result = pipeline.fit(data).transform(data)
```

```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_phs_bert", "en")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val sentence_embeddings = SentenceEmbeddings()
    .setInputCols(Array("sentence", "embeddings"))
    .setOutputCol("sentence_embeddings")
    .setPoolingStrategy("AVERAGE")

val classifier = ClassifierDLModel.pretrained("classifierdl_vaccine_sentiment", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "sentence_embeddings"))
    .setOutputCol("class")

val bert_clf_pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, embeddings, sentence_embeddings, classifier))

val data = Seq(Array("A little bright light for an otherwise dark week. Thanks researchers, and frontline workers. Onwards.", 
                     "People with a history of severe allergic reaction to any component of the vaccine should not take.", 
                     "43 million doses of vaccines administrated worldwide...Production capacity of CHINA to reach 4 b")).toDS.toDF("text")

val result = bert_clf_pipeline.fit(data).transform(data)
```


</div>

## Results

```bash
+-----------------------------------------------------------------------------------------------------+----------+
 |text                                                                                                 |class     |
 +-----------------------------------------------------------------------------------------------------+----------+
 |A little bright light for an otherwise dark week. Thanks researchers, and frontline workers. Onwards.|[positive]|
 |People with a history of severe allergic reaction to any component of the vaccine should not take.   |[negative]|
 |43 million doses of vaccines administrated worldwide...Production capacity of CHINA to reach 4 b     |[neutral] |
 +-----------------------------------------------------------------------------------------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_vaccine_sentiment|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|24.2 MB|

## References

Curated from several academic and in-house datasets.

## Benchmarking

```bash
       label  precision    recall  f1-score   support
     neutral       0.76      0.72      0.74      1008
    positive       0.80      0.79      0.80       966
    negative       0.76      0.81      0.78       916
    accuracy       -         -         0.77      2890
   macro-avg       0.77      0.77      0.77      2890
weighted-avg       0.77      0.77      0.77      2890
```
