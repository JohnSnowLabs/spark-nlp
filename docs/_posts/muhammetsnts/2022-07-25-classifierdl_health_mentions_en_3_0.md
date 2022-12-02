---
layout: model
title: Public Health Mention Classifier (PHS-BERT)
author: John Snow Labs
name: classifierdl_health_mentions
date: 2022-07-25
tags: [public_health, health, mention, en, licensed, classification]
task: Text Classification
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

This model is a [PHS-BERT](https://arxiv.org/abs/2204.04521) based classifier that can classify public health mentions in social media text. Mentions are classified into three labels about personal health situation, figurative mention and other mentions. More detailed information about classes as follows:

`health_mention`: The text contains a health mention that specifically indicating someone's health situation.  This means someone has a certain disease or symptoms including death. e.g.; *My PCR test is positive. I have a severe joint pain, mucsle pain and headache right now.*

`other_mention`: The text contains a health mention; however does not states a spesific person's situation. General health mentions like informative mentions, discussion about disease etc. e.g.; *Aluminum is a light metal that causes dementia and Alzheimer's disease.*

`figurative_mention`: The text mention specific disease or symptom but it is used metaphorically, does not contain health-related information. e.g.; *I don't wanna fall in love. If I ever did that, I think I'd have a heart attack.*

## Predicted Entities

`figurative_mention`, `other_mention`, `health_mention`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/classifierdl_health_mentions_en_4.0.0_3.0_1658759311177.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

classifierdl = ClassifierDLModel.pretrained('classifierdl_health_mentions', 'en', 'clinical/models')\
    .setInputCols(['sentence', 'token', 'sentence_embeddings'])\
    .setOutputCol('class')

clf_pipeline = Pipeline(
    stages = [
        document_assembler,
        tokenizer,
        bert_embeddings,
        embeddingsSentence,
        classifierdl
    ])

data = spark.createDataFrame([["I feel a bit drowsy & have a little blurred vision after taking an insulin."]]).toDF("text")
result = clf_pipeline.fit(data).transform(data)
result.select("text", "class.result").show(truncate=False)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased", "en")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("word_embeddings")

val sentence_embeddings = SentenceEmbeddings()
    .setInputCols(Array("sentence", "word_embeddings"))
    .setOutputCol("sentence_embeddings")
    .setPoolingStrategy("AVERAGE")

val classifier = ClassifierDLModel.pretrained("classifierdl_health_mentions", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "sentence_embeddings"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, embeddings, sentence_embeddings, classifier))

val data = Seq("I feel a bit drowsy & have a little blurred vision after taking an insulin.").toDF("text")
val result = pipeline.fit(data).transform(data) 
```
</div>

## Results

```bash
+---------------------------------------------------------------------------+----------------+
|text                                                                       |class           |
+---------------------------------------------------------------------------+----------------+
|I feel a bit drowsy & have a little blurred vision after taking an insulin.|[health_mention]|
+---------------------------------------------------------------------------+----------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_health_mentions|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|24.1 MB|

## References

Curated from several academic and in-house datasets.

## Benchmarking

```bash
                    precision    recall  f1-score   support
    health_mention       0.77      0.83      0.80      1375
     other_mention       0.84      0.81      0.83      2102
figurative_mention       0.79      0.78      0.79      1412
          accuracy       -         -         0.81      4889
         macro-avg       0.80      0.81      0.80      4889
      weighted-avg       0.81      0.81      0.81      4889
```
