--- 
layout: docs  
header: true  
seotitle:  
title: Text Preprocessing  
permalink: docs/en/tasks/text_preprocessing  
key: docs-tasks-text-preprocessing  
modify_date: "2024-10-05"  
show_nav: true  
sidebar:  
  nav: sparknlp  
---

**Text preprocessing** is a critical step in **Natural Language Processing (NLP)** that converts raw, unstructured text into a clean and analyzable form. It typically includes operations such as **tokenization**, **lowercasing**, **stopword removal**, **lemmatization or stemming**, and **handling of punctuation or special characters**. These steps reduce noise, ensure uniformity, and improve the performance of downstream NLP models.

## Key Preprocessing Steps

When preprocessing text, consider the following key steps along with the recommended Spark NLP annotators:

1. [`Tokenization:`](https://sparknlp.org/docs/en/annotators#tokenizer){:target="_blank"} Break text into smaller units (words, subwords, or sentences).
2. [`Spell Checking:`](https://sparknlp.org/docs/en/annotators#norvigsweeting-spellchecker){:target="_blank"} Correct misspelled words to improve accuracy in NLP tasks.
3. [`Normalization:`](https://sparknlp.org/docs/en/annotators#normalizer){:target="_blank"} Standardize text by converting to lowercase, expanding contractions, or removing accents.
4. [`Stopword Removal:`](https://sparknlp.org/docs/en/annotators#stopwordscleaner){:target="_blank"} Remove common, non-informative words (e.g., "the," "is," "and").
5. [`Lemmatization:`](https://sparknlp.org/docs/en/annotators#lemmatizer){:target="_blank"} Reduce words to their base form (e.g., "running" → "run").

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("tokens")

spellChecker = NorvigSweetingModel.pretrained() \
    .setInputCols(["tokens"]) \
    .setOutputCol("corrected")

normalizer = Normalizer() \
    .setInputCols(["corrected"]) \
    .setOutputCol("normalized")

stopwordsCleaner = StopWordsCleaner() \
    .setInputCols(["normalized"]) \
    .setOutputCol("cleanTokens")

lemmatizer = LemmatizerModel.pretrained() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("lemmas")

pipeline = Pipeline().setStages([
    documentAssembler, tokenizer, spellChecker, normalizer, stopwordsCleaner, lemmatizer
])

data = spark.createDataFrame([["Dr. Emily Johnson visited New York's Mount Sinai Hospital on September 21, 2023, to evaluate patients suffering from chronic migraines."]]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("tokens")

val spellChecker = NorvigSweetingModel.pretrained()
  .setInputCols("tokens")
  .setOutputCol("corrected")

val normalizer = new Normalizer()
  .setInputCols("corrected")
  .setOutputCol("normalized")

val stopwordsCleaner = new StopWordsCleaner()
  .setInputCols("normalized")
  .setOutputCol("cleanTokens")

val lemmatizer = LemmatizerModel.pretrained()
  .setInputCols("cleanTokens")
  .setOutputCol("lemmas")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler, tokenizer, spellChecker, normalizer, stopwordsCleaner, lemmatizer
))

val data = Seq("Dr. Emily Johnson visited New York's Mount Sinai Hospital on September 21, 2023, to evaluate patients suffering from chronic migraines.")
  .toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

```
</div>

<div class="tabs-box" markdown="1">
```
+---------+---------+----------+----------+---------+
|Token    |Corrected|Normalized|CleanToken|Lemma    |
+---------+---------+----------+----------+---------+
|Dr       |Dr       |Dr        |Dr        |Dr       |
|.        |.        |Emily     |Emily     |Emily    |
|Emily    |Emily    |Johnson   |Johnson   |Johnson  |
|Johnson  |Johnson  |visited   |visited   |visit    |
|visited  |visited  |New       |New       |New      |
|New      |New      |Yorks     |Yorks     |Yorks    |
|York's   |Yorks    |Mount     |Mount     |Mount    |
|Mount    |Mount    |Sinai     |Sinai     |Sinai    |
|Sinai    |Sinai    |Hospital  |Hospital  |Hospital |
|Hospital |Hospital |on        |September |September|
|on       |on       |September |evaluate  |evaluate |
|September|September|to        |patients  |patient  |
|21       |21       |evaluate  |suffering |suffer   |
|,        |,        |patients  |chronic   |chronic  |
|2023     |2023     |suffering |migraines |migraine |
|,        |,        |from      |NULL      |NULL     |
|to       |to       |chronic   |NULL      |NULL     |
|evaluate |evaluate |migraines |NULL      |NULL     |
|patients |patients |NULL      |NULL      |NULL     |
|suffering|suffering|NULL      |NULL      |NULL     |
+---------+---------+----------+----------+---------+
```
</div>

## Try Real-Time Demos!

If you want to see text preprocessing in real-time, check out our interactive demos:

- **[Text Preprocessing with Spark NLP](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-text-preprocessing){:target="_blank"}**
- **[Stopwords Removing with Spark NLP](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-stop-words-removal){:target="_blank"}**

## Useful Resources

Want to learn more about text preprocessing with Spark NLP? Explore the following resources:

**Articles and Guides**
- *[Text cleaning: removing stopwords from text with Spark NLP](https://www.johnsnowlabs.com/text-cleaning-removing-stopwords-from-text-with-spark-nlp/){:target="_blank"}*
- *[Unleashing the Power of Text Tokenization with Spark NLP](https://www.johnsnowlabs.com/unleashing-the-power-of-text-tokenization-with-spark-nlp/){:target="_blank"}*
- *[Tokenizing Asian texts into words with word segmentation models in Spark NLP](https://medium.com/john-snow-labs/tokenizing-asian-texts-into-words-with-word-segmentation-models-42e04d8e03da){:target="_blank"}*
- *[Text Cleaning: Standard Text Normalization with Spark NLP](https://www.johnsnowlabs.com/text-cleaning-standard-text-normalization-with-spark-nlp/){:target="_blank"}*
- *[Boost Your NLP Results with Spark NLP Stemming and Lemmatizing Techniques](https://www.johnsnowlabs.com/boost-your-nlp-results-with-spark-nlp-stemming-and-lemmatizing-techniques/){:target="_blank"}*
- *[Sample Text Data Preprocessing Implementation In SparkNLP](https://ahmetemin-tek.medium.com/sample-text-data-preprocessing-implementation-in-sparknlp-5de53085fed6){:target="_blank"}*

**Notebooks** 
- *[Text Preprocessing with SparkNLP Annotators Transformers](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb){:target="_blank"}*
- *[Text_Preprocessing_with_SparkNLP](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/Text_Preprocessing_with_SparkNLP.ipynb){:target="_blank"}*
- *[Word Stemming with Stemmer](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/stemmer/Word_Stemming_with_Stemmer.ipynb){:target="_blank"}*
- *[Document Normalizer](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/document-normalizer/document_normalizer_notebook.ipynb){:target="_blank"}*
- *[Cleaning Stop Words](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/stop-words/StopWordsCleaner.ipynb){:target="_blank"}*
