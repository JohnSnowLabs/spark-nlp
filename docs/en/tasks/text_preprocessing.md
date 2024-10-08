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

**Text Preprocessing** is the foundational task of cleaning and transforming raw text data into a structured format that can be used in NLP tasks. It involves a series of steps to normalize text, remove noise, and prepare it for deeper analysis. Spark NLP provides a range of tools for efficient and scalable text preprocessing.

## Key Preprocessing Steps

When preprocessing text, consider the following key steps along with the recommended Spark NLP annotators:

1. [`Tokenization:`](https://sparknlp.org/docs/en/annotators#tokenizer){:target="_blank"} Break text into smaller units (words, subwords, or sentences).
2. [`Spell Checking:`](https://sparknlp.org/docs/en/annotators#norvigsweeting-spellchecker){:target="_blank"} Correct misspelled words to improve accuracy in NLP tasks.
3. [`Normalization:`](https://sparknlp.org/docs/en/annotators#normalizer){:target="_blank"} Standardize text by converting to lowercase, expanding contractions, or removing accents.
4. [`Stopword Removal:`](https://sparknlp.org/docs/en/annotators#stopwordscleaner){:target="_blank"} Remove common, non-informative words (e.g., "the," "is," "and").
5. [`Lemmatization:`](https://sparknlp.org/docs/en/annotators#lemmatizer){:target="_blank"} Reduce words to their base form (e.g., "running" → "run").

These steps and annotators will help ensure your text data is clean, consistent, and ready for analysis.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# Document Assembler: Converts input text into a suitable format for NLP processing
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

# Tokenizer: Splits text into individual tokens (words)
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("tokens")

# SpellChecker: Corrects misspelled words
spellChecker = NorvigSweetingModel.pretrained() \
    .setInputCols(["tokens"]) \
    .setOutputCol("corrected")

# Normalizer: Cleans and standardizes text data
normalizer = Normalizer() \
    .setInputCols(["corrected"]) \
    .setOutputCol("normalized")

# StopWordsCleaner: Removes stopwords
stopwordsCleaner = StopWordsCleaner() \
    .setInputCols(["normalized"]) \
    .setOutputCol("cleanTokens")

# Lemmatizer: Reduces words to their base form
lemmatizer = LemmatizerModel.pretrained() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("lemmas")

# Pipeline: Assembles the document assembler and preprocessing stages
pipeline = Pipeline().setStages([
    documentAssembler, tokenizer, spellChecker, normalizer, stopwordsCleaner, lemmatizer
])

# Input Data: A small example dataset is created and converted to a DataFrame
data = spark.createDataFrame([["Text preprocessing is essential in NLP!"]]).toDF("text")

# Running the Pipeline: Fits the pipeline to the data and preprocesses the text
result = pipeline.fit(data).transform(data)

# Output: Displays the processed tokens and lemmas
result.select("lemmas.result").show(truncate=False)

+----------------------------------------------------+
|lemmas.result                                       |
+----------------------------------------------------+
|[text, preprocess, essential, in, NLP]              |
+----------------------------------------------------+
```
```scala
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

// Document Assembler: Converts input text into a suitable format for NLP processing
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

// Tokenizer: Splits text into individual tokens (words)
val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("tokens")

// SpellChecker: Corrects misspelled words
val spellChecker = NorvigSweetingModel.pretrained()
  .setInputCols(Array("tokens"))
  .setOutputCol("corrected")

// Normalizer: Cleans and standardizes text data
val normalizer = new Normalizer()
  .setInputCols(Array("corrected"))
  .setOutputCol("normalized")

// StopWordsCleaner: Removes stopwords
val stopwordsCleaner = new StopWordsCleaner()
  .setInputCols(Array("normalized"))
  .setOutputCol("cleanTokens")

// Lemmatizer: Reduces words to their base form
val lemmatizer = LemmatizerModel.pretrained()
  .setInputCols(Array("cleanTokens"))
  .setOutputCol("lemmas")

// Pipeline: Assembles the document assembler and preprocessing stages
val pipeline = new Pipeline().setStages(Array(
  documentAssembler, tokenizer, spellChecker, normalizer, stopwordsCleaner, lemmatizer
))

// Input Data: A small example dataset is created and converted to a DataFrame
val data = Seq("Text preprocessing is essential in NLP!").toDF("text")

// Running the Pipeline: Fits the pipeline to the data and preprocesses the text
val result = pipeline.fit(data).transform(data)

// Display the results
result.select("lemmas.result").show(false)

+----------------------------------------------------+
|result                                              |
+----------------------------------------------------+
|[text, preprocess, essential, in, NLP]              |
+----------------------------------------------------+
```
</div>

## Try Real-Time Demos!

If you want to see text preprocessing in real-time, check out our interactive demos:

- **[Text Preprocessing with Spark NLP](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-text-preprocessing){:target="_blank"}** – Explore how Spark NLP preprocesses raw text data.
- **[Stopwords Removing with Spark NLP](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-stop-words-removal){:target="_blank"}** – Explore how Spark NLP removes stop words from text.

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
