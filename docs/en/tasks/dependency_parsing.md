--- 
layout: docs  
header: true  
seotitle:  
title: Dependency Parsing  
permalink: docs/en/tasks/dependency_parsing  
key: docs-tasks-dependency-parsing  
modify_date: "2024-09-28"  
show_nav: true  
sidebar:  
  nav: sparknlp  
---  

**Dependency Parsing** is a syntactic analysis task that focuses on the grammatical structure of sentences. It identifies the dependencies between words, showcasing how they relate in terms of grammar. Spark NLP provides advanced dependency parsing models that can accurately analyze sentence structures, enabling various applications in natural language processing.

Dependency parsing models process input sentences and generate a structured representation of word relationships. Common use cases include:

- **Grammatical Analysis:** Understanding the grammatical structure of sentences for better comprehension.
- **Information Extraction:** Identifying key relationships and entities in sentences for tasks like knowledge graph construction.

By using Spark NLP dependency parsing models, you can build efficient systems to analyze and understand sentence structures accurately.

## Picking a Model

When selecting a dependency parsing model, consider factors such as the **language of the text** and the **complexity of sentence structures**. Some models may be optimized for specific languages or types of text. Evaluate whether you need **detailed syntactic parsing** or a more **general analysis** based on your application.

Explore the available dependency parsing models at [Spark NLP Models](https://sparknlp.org/models) to find the one that best fits your requirements.

#### Recommended Models for Dependency Parsing Tasks

- **General Dependency Parsing:** Consider models such as [`dependency_conllu_en_3_0`](https://sparknlp.org/2022/06/29/dependency_conllu_en_3_0.html){:target="_blank"} for analyzing English sentences. You can also explore language-specific models tailored for non-English languages.

Choosing the appropriate model ensures you produce accurate syntactic structures that suit your specific language and use case.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.ml import Pipeline

# Document Assembler: Converts raw text into a document format suitable for processing
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

# Sentence Detector: Splits text into individual sentences
sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

# Tokenizer: Breaks sentences into tokens (words)
tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

# Part-of-Speech Tagger: Tags each token with its respective POS (pretrained model)
posTagger = PerceptronModel.pretrained() \
    .setInputCols(["token", "sentence"]) \
    .setOutputCol("pos")

# Dependency Parser: Analyzes the grammatical structure of a sentence
dependencyParser = DependencyParserModel.pretrained() \
    .setInputCols(["sentence", "pos", "token"]) \
    .setOutputCol("dependency")

# Typed Dependency Parser: Assigns typed labels to the dependencies
typedDependencyParser = TypedDependencyParserModel.pretrained() \
    .setInputCols(["token", "pos", "dependency"]) \
    .setOutputCol("labdep")

# Create a pipeline that includes all the stages
pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector, 
    tokenizer, 
    posTagger, 
    dependencyParser, 
    typedDependencyParser
])

# Sample input data (a DataFrame with one text example)
data = {"text": ["Dependencies represent relationships between words in a sentence."]}
df = spark.createDataFrame(data)

# Run the pipeline on the input data
result = pipeline.fit(df).transform(df)

# Show the dependency parsing results
result.select("dependency.result").show(truncate=False)

+---------------------------------------------------------------------------------+
|result                                                                           |
+---------------------------------------------------------------------------------+
|[ROOT, Dependencies, represents, words, relationships, Sentence, Sentence, words]|
+---------------------------------------------------------------------------------+
```
```scala
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

// Document Assembler: Converts raw text into a document format for NLP processing
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

// Sentence Detector: Splits the input text into individual sentences
val sentenceDetector = new SentenceDetector()
  .setInputCols(Array("document"))
  .setOutputCol("sentence")

// Tokenizer: Breaks sentences into individual tokens (words)
val tokenizer = new Tokenizer()
  .setInputCols(Array("sentence"))
  .setOutputCol("token")

// Part-of-Speech Tagger: Tags each token with its respective part of speech (pretrained model)
val posTagger = PerceptronModel.pretrained()
  .setInputCols(Array("token", "sentence"))
  .setOutputCol("pos")

// Dependency Parser: Analyzes the grammatical structure of the sentence
val dependencyParser = DependencyParserModel.pretrained()
  .setInputCols(Array("sentence", "pos", "token"))
  .setOutputCol("dependency")

// Typed Dependency Parser: Assigns typed labels to the dependencies
val typedDependencyParser = TypedDependencyParserModel.pretrained()
  .setInputCols(Array("token", "pos", "dependency"))
  .setOutputCol("labdep")

// Create a pipeline that includes all stages
val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  posTagger,
  dependencyParser,
  typedDependencyParser
))

// Sample input data (a DataFrame with one text example)
val df = Seq("Dependencies represent relationships between words in a Sentence").toDF("text")

// Run the pipeline on the input data
val result = pipeline.fit(df).transform(df)

// Show the dependency parsing results
result.select("dependency.result").show(truncate = false)

+---------------------------------------------------------------------------------+
|result                                                                           |
+---------------------------------------------------------------------------------+
|[ROOT, Dependencies, represents, words, relationships, Sentence, Sentence, words]|
+---------------------------------------------------------------------------------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of dependency parsing models in real time, visit our interactive demos:

- **[Grammar Analysis & Dependency Parsing](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-grammar-analysis-and-dependency-parsing){:target="_blank"}** – An interactive demo to visualize dependencies in sentences.

## Useful Resources

Want to dive deeper into dependency parsing with Spark NLP? Here are some curated resources to help you get started and explore further:

**Articles and Guides**
- *[Mastering Dependency Parsing with Spark NLP and Python](https://www.johnsnowlabs.com/supercharge-your-nlp-skills-mastering-dependency-parsing-with-spark-nlp-and-python/){:target="_blank"}*

**Notebooks**
- *[Extract Part of speech tags and perform dependency parsing on a text](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb#scrollTo=syePZ-1gYyj3){:target="_blank"}*
- *[Typed Dependency Parsing with NLU.](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/nlu/colab/component_examples/dependency_parsing/NLU_typed_dependency_parsing_example.ipynb){:target="_blank"}*
