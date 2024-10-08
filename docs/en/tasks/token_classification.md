---
layout: docs  
header: true  
seotitle:  
title: Token Classification  
permalink: docs/en/tasks/token_classification  
key: docs-tasks-token-classification  
modify_date: "2024-09-26"  
show_nav: true  
sidebar:  
  nav: sparknlp  
---

**Token classification** is the task of assigning a **label** to each token (word or sub-word) in a given text sequence. It is fundamental in various *natural language processing (NLP)* tasks like named entity recognition (NER), part-of-speech tagging (POS), and more. Spark NLP provides state of the art solutions to tackle token classification challenges effectively, helping you analyze and label individual tokens in a document.

Token classification involves processing text at a granular level, labeling each token for its role or entity. Typical use cases include:

- **Named Entity Recognition (NER):** Identifying proper names, locations, organizations, etc., within text.
- **Part-of-Speech Tagging (POS):** Labeling each token with its grammatical category (e.g., noun, verb, adjective).

By utilizing token classification, organizations can enhance their ability to extract detailed insights from text data, enabling applications like information extraction, text annotation, and more.

<div style="text-align: center;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/B3xB9gaBosw?si=hDgXLUoduQkkodPN&amp;start=258" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

## Picking a Model

When selecting a model for token classification, it's important to consider various factors that impact performance. First, analyze the **type of entities or tags** you want to classify (e.g., named entities, parts of speech). Determine if your task requires **fine-grained tagging** (such as multiple types of named entities) or a simpler tag set.

Next, assess the **complexity of your data**—does it involve formal text like news articles, or informal text like social media posts? **Model performance metrics** (e.g., precision, recall, F1 score) are also key to determining whether a model is suitable. Lastly, evaluate your **computational resources**, as more complex models like BERT may require greater memory and processing power.

You can explore and select models for your token classification tasks at [Spark NLP Models](https://sparknlp.org/models), where you'll find various models for specific datasets and challenges.

#### Recommended Models for Specific Token Classification Tasks

- **Named Entity Recognition (NER):** Use models like [`bert-base-NER`](https://sparknlp.org/2022/05/09/bert_ner_bert_base_NER_en_3_0.html){:target="_blank"} and [`xlm-roberta-large-finetuned-conll03-english`](https://sparknlp.org/2022/08/14/xlmroberta_ner_large_finetuned_conll03_english_xx_3_0.html){:target="_blank"} for general-purpose NER tasks.
- **Part-of-Speech Tagging (POS):** For POS tagging, consider using models such as [`pos_anc`](https://sparknlp.org/2021/03/05/pos_anc.html){:target="_blank"}.
- **Healthcare NER:** For clinical texts, [`ner_jsl`](https://nlp.johnsnowlabs.com/2022/10/19/ner_jsl_en.html){:target="_blank"} and [`pos_clinical`](https://sparknlp.org/2023/02/17/ner_jsl_en.html){:target="_blank"} is tailored for extracting medical entities.

If existing models do not meet your requirements, you can train your own custom model using the [Spark NLP Training Documentation](https://sparknlp.org/docs/en/training).

By selecting the appropriate model, you can optimize token classification performance for your specific NLP tasks.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# Assembling the document from the input text
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

# Tokenizing the text
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

# Loading a pre-trained sequence classification model
# You can replace `BertForTokenClassification.pretrained()` with your selected model and the transformer it's based on
# For example: XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_large_finetuned_conll03_english","xx")
tokenClassifier = BertForTokenClassification.pretrained() \
    .setInputCols(["token", "document"]) \
    .setOutputCol("label") \
    .setCaseSensitive(True)

# Defining the pipeline with document assembler, tokenizer, and classifier
pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    tokenClassifier
])

# Creating a sample DataFrame
data = spark.createDataFrame([["John Lenon was born in London and lived in Paris. My name is Sarah and I live in London"]]).toDF("text")

# Fitting the pipeline and transforming the data
result = pipeline.fit(data).transform(data)

# Showing the results
result.select("label.result").show(truncate=False)

<!-- 
+------------------------------------------------------------------------------------+
|result                                                                              |
+------------------------------------------------------------------------------------+
|[B-PER, I-PER, O, O, O, B-LOC, O, O, O, B-LOC, O, O, O, O, B-PER, O, O, O, O, B-LOC]|
+------------------------------------------------------------------------------------+
-->
```

```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

// Step 1: Assembling the document from the input text
// Converts the input 'text' column into a 'document' column, required for NLP tasks
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

// Step 2: Tokenizing the text
// Splits the 'document' column into tokens (words), creating the 'token' column
val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

// Step 3: Loading a pre-trained BERT model for token classification
// Applies a pre-trained BERT model for Named Entity Recognition (NER) to classify tokens
// `BertForTokenClassification.pretrained()` loads the model, and `setInputCols` defines the input columns
val tokenClassifier = BertForTokenClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)

// Step 4: Defining the pipeline
// The pipeline stages are document assembler, tokenizer, and token classifier
val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  tokenClassifier
))

// Step 5: Creating a sample DataFrame
// Creates a DataFrame with a sample sentence that will be processed by the pipeline
val data = Seq("John Lenon was born in London and lived in Paris. My name is Sarah and I live in London").toDF("text")

// Step 6: Fitting the pipeline and transforming the data
// The pipeline is fitted on the input data, then it performs the transformation to generate token labels
val result = pipeline.fit(data).transform(data)

// Step 7: Showing the results
// Displays the 'label.result' column, which contains the Named Entity Recognition (NER) labels for each token
result.select("label.result").show(false)

// Output:
// +------------------------------------------------------------------------------------+
// |result                                                                              |
// +------------------------------------------------------------------------------------+
// |[B-PER, I-PER, O, O, O, B-LOC, O, O, O, B-LOC, O, O, O, O, B-PER, O, O, O, O, B-LOC]|
// +------------------------------------------------------------------------------------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of text classification models in real time, visit our interactive demos:

- **[BERT Annotators Demo](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-bert-annotators){:target="_blank"}** – A live demo where you can try your inputs on classification models on the go.
- **[Named Entity Recognition (NER)](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-named-entity-recognition){:target="_blank"}** – A live demo where you can try your inputs on NER models on the go.
- **[POS Tagging](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-POS-tagging){:target="_blank"}** – A live demo where you can try your inputs on preception models on the go.
- **[Recognize Entities - Live Demos & Notebooks](https://sparknlp.org/recognize_entitie){:target="_blank"}** – An interactive demo for Recognizing Entities in text

## Useful Resources

Want to dive deeper into text classification with Spark NLP? Here are some curated resources to help you get started and explore further:

**Articles and Guides**
- *[Named Entity Recognition (NER) with BERT in Spark NLP](https://www.johnsnowlabs.com/named-entity-recognition-ner-with-bert-in-spark-nlp/){:target="_blank"}*
- *[The Ultimate Guide to Rule-based Entity Recognition with Spark NLP](https://www.johnsnowlabs.com/rule-based-entity-recognition-with-spark-nlp/){:target="_blank"}*
- *[In-Depth Comparison of Spark NLP for Healthcare and ChatGPT on Clinical Named Entity Recognition](https://www.johnsnowlabs.com/in-depth-comparison-of-spark-nlp-for-healthcare-and-chatgpt-on-clinical-named-entity-recognition/){:target="_blank"}*

**Notebooks**
- *[Transformers for Token Classification in Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/4.3_Transformers_for_Token_Classification_in_Spark_NLP.ipynb){:target="_blank"}*

**Training Scripts**
- *[Training Named Entity Recognition (NER) Deep-Learning models](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/training/english/dl-ner){:target="_blank"}*
- *[Training Conditional Random Fields (CRF) Named Entity Recognition (NER) models](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/crf-ner/ner_dl_crf.ipynb){:target="_blank"}*