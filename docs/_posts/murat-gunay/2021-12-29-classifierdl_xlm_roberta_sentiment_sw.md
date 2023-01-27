---
layout: model
title: Sentiment Analysis of Swahili texts
author: John Snow Labs
name: classifierdl_xlm_roberta_sentiment
date: 2021-12-29
tags: [swahili, sentiment, sw, open_source]
task: Sentiment Analysis
language: sw
edition: Spark NLP 3.3.4
spark_version: 3.0
supported: true
annotator: ClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model identifies positive or negative sentiments in Swahili texts.

## Predicted Entities

`Negative`, `Positive`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_SW/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_SW.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_xlm_roberta_sentiment_sw_3.3.4_3.0_1640766370034.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classifierdl_xlm_roberta_sentiment_sw_3.3.4_3.0_1640766370034.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

tokenizer = Tokenizer() \
      .setInputCols(["document"]) \
      .setOutputCol("token")
    
normalizer = Normalizer() \
      .setInputCols(["token"]) \
      .setOutputCol("normalized")

stopwords_cleaner = StopWordsCleaner.pretrained("stopwords_sw", "sw") \
        .setInputCols(["normalized"]) \
        .setOutputCol("cleanTokens")\
        .setCaseSensitive(False)

embeddings = XlmRoBertaEmbeddings.pretrained("xlm_roberta_base_finetuned_swahili", "sw")\
    .setInputCols(["document", "cleanTokens"])\
    .setOutputCol("embeddings")

embeddingsSentence = SentenceEmbeddings() \
      .setInputCols(["document", "embeddings"]) \
      .setOutputCol("sentence_embeddings") \
      .setPoolingStrategy("AVERAGE")

sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_xlm_roberta_sentiment", "sw") \
  .setInputCols(["document", "sentence_embeddings"]) \
  .setOutputCol("class")

sw_pipeline = Pipeline(stages=[document_assembler, tokenizer, normalizer, stopwords_cleaner, embeddings, embeddingsSentence, sentimentClassifier])

light_pipeline = LightPipeline(sw_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

result1 = light_pipeline.annotate("Hadithi yenyewe ni ya kutabirika tu na ya uvivu.")

result2 = light_pipeline.annotate("Mtandao wa kushangaza wa 4G katika mji wa Mombasa pamoja na mipango nzuri sana na ya bei rahisi.")

print(result1["class"], result2["class"], sep = "\n")
```
```scala
val document_assembler = DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

val tokenizer = Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")
    
val normalizer = Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")

val stopwords_cleaner = StopWordsCleaner.pretrained("stopwords_sw", "sw")
        .setInputCols(Array("normalized"))
        .setOutputCol("cleanTokens")
        .setCaseSensitive(False)

val embeddings = XlmRoBertaEmbeddings.pretrained("xlm_roberta_base_finetuned_swahili", "sw")
    .setInputCols(Array("document", "cleanTokens"))
    .setOutputCol("embeddings")

val embeddingsSentence = SentenceEmbeddings()
      .setInputCols(Array("document", "embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

val sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_xlm_roberta_sentiment", "sw")
  .setInputCols(Array("document", "sentence_embeddings"))
  .setOutputCol("class")

val sw_sentiment_pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, normalizer, stopwords_cleaner, embeddings, embeddingsSentence, sentimentClassifier))

val light_pipeline = LightPipeline(sw_sentiment_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

val result1 = light_pipeline.annotate("Hadithi yenyewe ni ya kutabirika tu na ya uvivu.")

val result2 = light_pipeline.annotate("Mtandao wa kushangaza wa 4G katika mji wa Mombasa pamoja na mipango nzuri sana na ya bei rahisi.")
```
</div>

## Results

```bash
['Negative']
['Positive']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_xlm_roberta_sentiment|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|sw|
|Size:|23.0 MB|

## Data Source

[https://github.com/Jinamizi/Swahili-sentiment-analysis](https://github.com/Jinamizi/Swahili-sentiment-analysis)

## Benchmarking

```bash
       label  precision    recall  f1-score   support
    Negative       0.79      0.84      0.81        85
    Positive       0.86      0.82      0.84       103
    accuracy          -         -      0.82       188
   macro-avg       0.82      0.83      0.82       188
weighted-avg       0.83      0.82      0.82       188
```
