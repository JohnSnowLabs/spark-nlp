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

**Text classification** is a natural language processing task where entire pieces of text, such as sentences, paragraphs, or documents, are assigned *predefined labels*. Common subtasks include **sentiment analysis** and **topic classification**. For instance, *sentiment analysis* models can determine whether a review is *positive*, *negative*, or *neutral*, while *topic classification* models can categorize news articles into areas like **politics**, **sports**, or **technology**.

<div style="text-align: center;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/B3xB9gaBosw?si=BDII1NUUE2eSkME6&amp;start=245" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

## Picking a Model

When picking a model for text classification, think about what you’re trying to do—like detecting spam (binary), analyzing sentiment across positive, neutral, and negative (multiclass), or tagging a news article with several topics at once (multi-label). If you only have a small dataset, lightweight options such as **DistilBERT** are quick to train and deploy, while larger transformers like **BERT** or **RoBERTa** generally give better accuracy when fine-tuned on enough data. For specialized fields, domain-trained models like [**BioBERT**](https://sparknlp.org/models?q=biobert&sort=downloads&task=Text+Classification&type=model&annotator=BertForSequenceClassification) for biomedical research or [**FinBERT**](https://sparknlp.org/models?q=FinBERT&sort=downloads&task=Text+Classification&type=model&annotator=BertForSequenceClassification) for finance usually outperform general-purpose ones. Finally, keep in mind practical constraints—how much compute you have, whether you need real-time predictions, how important explainability is, and what balance you want between speed, cost, and accuracy.

To explore and select from a variety of models, visit [Spark NLP Models](https://sparknlp.org/models), where you can find models tailored for different tasks and datasets.

#### Recommended Models for Specific Text Classification Tasks
- **Sentiment Analysis:** Use models specifically designed for sentiment detection, such as [`distilbert_sequence_classifier_sst2`](https://sparknlp.org/2021/11/21/distilbert_sequence_classifier_sst2_en.html){:target="_blank"}.
- **News Categorization:** Models like [`distilroberta-finetuned-financial-news-sentiment-analysis`](https://sparknlp.org/2023/11/29/roberta_sequence_classifier_distilroberta_finetuned_financial_news_sentiment_analysis_en.html){:target="_blank"} are ideal for classifying news articles into relevant categories.
- **Review Analysis:** For product reviews, consider using [`distilbert_base_uncased_finetuned_sentiment_amazon`](https://sparknlp.org/2023/11/18/distilbert_base_uncased_finetuned_sentiment_amazon_en.html){:target="_blank"} for more nuanced insights.

If you have specific needs that are not covered by existing models, you can train your own model tailored to your unique requirements. Follow the guidelines provided in the [Spark NLP Training Documentation](https://sparknlp.org/docs/en/training) to get started.

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
    .setOutputCol("token")

sequenceClassifier = BertForSequenceClassification.pretrained() \
    .setInputCols(["token", "document"]) \
    .setOutputCol("label") \
    .setCaseSensitive(True)

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    sequenceClassifier
])

data = spark.createDataFrame([
    ("I loved this movie when I was a child.",),
    ("It was pretty boring.",)
]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.select("text", "label.result").show(truncate=False)

```

```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val sequenceClassifier = BertForSequenceClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  sequenceClassifier
))

val data = Seq("I loved this movie when I was a child.", "It was pretty boring.").toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("text", "label.result").show(truncate=False)

```
</div>

<div class="tabs-box" markdown="1">
```
+--------------------------------------+------+
|text                                  |result|
+--------------------------------------+------+
|I loved this movie when I was a child.|[pos] |
|It was pretty boring.                 |[neg] |
+--------------------------------------+------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of text classification models in real time, visit our interactive demos:

- **[Sentiment & Emotion Detection Demo](https://nlp.johnsnowlabs.com/detect_sentiment_emotion){:target="_blank"}**
- **[BERT Annotators Demo](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-bert-annotators){:target="_blank"}**

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