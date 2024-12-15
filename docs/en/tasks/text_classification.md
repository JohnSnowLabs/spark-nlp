---
layout: docs
header: true
seotitle:
title: Text Classification
permalink: docs/en/tasks/text_classification
key: docs-tasks-text-classification
modify_date: "2024-09-26"
show_nav: true
sidebar:
  nav: sparknlp
---

**Text classification** is the process of assigning a **category** or **label** to a piece of text, such as an email, tweet, or review. It plays a crucial role in *natural language processing (NLP)*, where it is used to automatically organize text into predefined categories. Spark NLP provides various solutions to address text classification challenges effectively.

In this context, text classification involves analyzing a document's content to categorize it into one or more predefined groups. Common use cases include:

- Organizing news articles into categories like **politics**, **sports**, **entertainment**, or **technology**.
- Conducting sentiment analysis, where customer reviews of products or services are classified as **positive**, **negative**, or **neutral**.

By leveraging text classification, organizations can enhance their ability to process and understand large volumes of text data efficiently.

<div style="text-align: center;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/B3xB9gaBosw?si=BDII1NUUE2eSkME6&amp;start=245" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

## Picking a Model

When selecting a model for text classification, it’s crucial to evaluate several factors to ensure optimal performance for your specific use case. Start by analyzing the **nature of your data**, considering whether it is formal or informal and its length (e.g., tweets vs. reviews). Determine if your task requires **binary classification** (like spam detection) or **multiclass classification** (such as categorizing news topics), as some models excel in specific scenarios.

Next, assess the **model complexity**; simpler models like Logistic Regression work well for straightforward tasks, while more complex models like BERT are suited for nuanced understanding. Consider the **availability of labeled data**—larger datasets allow for training sophisticated models, whereas smaller datasets may benefit from pre-trained options. Define key **performance metrics** (e.g., accuracy, F1 score) to inform your choice, and ensure the model's interpretability meets your requirements. Finally, account for **resource constraints**, as advanced models will demand more memory and processing power.

To explore and select from a variety of models, visit [Spark NLP Models](https://sparknlp.org/models), where you can find models tailored for different tasks and datasets.


#### Recommended Models for Specific Text Classification Tasks
- **Sentiment Analysis:** Use models specifically designed for sentiment detection, such as [`distilbert_sequence_classifier_sst2`](https://sparknlp.org/2021/11/21/distilbert_sequence_classifier_sst2_en.html){:target="_blank"}.
- **News Categorization:** Models like [`distilroberta-finetuned-financial-news-sentiment-analysis`](https://sparknlp.org/2023/11/29/roberta_sequence_classifier_distilroberta_finetuned_financial_news_sentiment_analysis_en.html){:target="_blank"} are ideal for classifying news articles into relevant categories.
- **Review Analysis:** For product reviews, consider using [`distilbert_base_uncased_finetuned_sentiment_amazon`](https://sparknlp.org/2023/11/18/distilbert_base_uncased_finetuned_sentiment_amazon_en.html){:target="_blank"} for more nuanced insights.

If you have specific needs that are not covered by existing models, you can train your own model tailored to your unique requirements. Follow the guidelines provided in the [Spark NLP Training Documentation](https://sparknlp.org/docs/en/training) to get started on creating and training a model suited for your text classification task.

By thoughtfully considering these factors and using the right models, you can enhance your NLP applications significantly.

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
# You can replace `BertForSequenceClassification.pretrained()` with your selected model 
# For example: BertForSequenceClassification.pretrained("distilbert_sequence_classifier_sst2", "en")
sequenceClassifier = BertForSequenceClassification.pretrained() \
    .setInputCols(["token", "document"]) \
    .setOutputCol("label") \
    .setCaseSensitive(True)

# Defining the pipeline with document assembler, tokenizer, and classifier
pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    sequenceClassifier
])

# Creating a sample DataFrame
data = spark.createDataFrame([["I loved this movie when I was a child.", "It was pretty boring."]]).toDF("text")

# Fitting the pipeline and transforming the data
result = pipeline.fit(data).transform(data)

# Showing the classification result
result.select("label.result").show(truncate=False)

+------+
|result|
+------+
|[pos] |
|[neg] |
+------+
```

```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

// Step 1: Convert raw text into document format
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

// Step 2: Tokenize the document into words
val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

// Step 3: Load a pre-trained BERT model for sequence classification
val sequenceClassifier = BertForSequenceClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)

// Step 4: Define the pipeline with stages for document assembly, tokenization, and classification
val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  sequenceClassifier
))

// Step 5: Create sample data and apply the pipeline
val data = Seq("I loved this movie when I was a child.", "It was pretty boring.").toDF("text")
val result = pipeline.fit(data).transform(data)

// Step 6: Show the classification results
result.select("label.result").show(false)

+------+
|result|
+------+
|[pos] |
|[neg] |
+------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of text classification models in real time, visit our interactive demos:

- **[BERT Annotators Demo](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-bert-annotators){:target="_blank"}** – A live demo where you can try your inputs on text classification models on the go.
- **[Sentiment & Emotion Detection Demo](https://nlp.johnsnowlabs.com/detect_sentiment_emotion){:target="_blank"}** – An interactive demo for sentiment and emotion detection.

## Useful Resources

Want to dive deeper into text classification with Spark NLP? Here are some curated resources to help you get started and explore further:

**Articles and Guides**
- *[Mastering Text Classification with Spark NLP](https://www.johnsnowlabs.com/mastering-text-classification-with-spark-nlp/){:target="_blank"}*
- *[Unlocking the Power of Sentiment Analysis with Deep Learning](https://www.johnsnowlabs.com/unlocking-the-power-of-sentiment-analysis-with-deep-learning/){:target="_blank"}*
- *[Sentiment Analysis with Spark NLP without Machine Learning](https://www.johnsnowlabs.com/sentiment-analysis-with-spark-nlp-without-machine-learning/){:target="_blank"}*
- *[Financial Sentiment Analysis Using SparkNLP Achieving 95% Accuracy](https://medium.com/spark-nlp/financial-sentiment-analysis-using-sparknlp-achieving-95-accuracy-e2df27744617){:target="_blank"}*

**Notebooks**
- *[Text Classification with ClassifierDL](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb){:target="_blank"}*

**Training Scripts**
- *[Training Multi-class Text and Sentiment Classification models](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/training/english/classification){:target="_blank"}*
- *[Training a text classification model with INSTRUCTOR Embeddings](https://medium.com/spark-nlp/training-a-text-classification-model-with-instructor-embeddings-1a29e8c8792b){:target="_blank"}*