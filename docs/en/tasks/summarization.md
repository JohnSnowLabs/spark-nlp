---
layout: docs  
header: true  
seotitle:  
title: Summarization  
permalink: docs/en/tasks/summarization  
key: docs-tasks-summarization  
modify_date: "2024-09-28"  
show_nav: true  
sidebar:  
  nav: sparknlp  
---

**Summarization** is a natural language processing task where models create a shorter version of a text while preserving its key information. Depending on the approach, models may use **extractive summarization**, which selects important sentences or phrases directly from the source, or **abstractive summarization**, which generates entirely new sentences that rephrase the original content. For example, given a passage about the Eiffel Tower’s height and history, a summarization model might produce *“The tower is 324 metres tall, about the height of an 81-storey building, and was the first structure to reach 300 metres.”*

This task is especially valuable for quickly processing large amounts of text in areas like **research paper summarization**, **news aggregation**, **financial reports**, and **legal documents**.

## Picking a Model  

The choice of model for summarization depends on whether the goal is extractive or abstractive. For **extractive summarization**, transformer-based classifiers like **BERTSUM** or lightweight variants such as **DistilBERT** can effectively identify the most important sentences to keep. For **abstractive summarization**, encoder–decoder architectures such as **BART** and **T5** are strong general-purpose options, while more recent families like **LLaMA 2** have shown strong performance when adapted for summarization tasks. In **domain-specific contexts** such as biomedical, legal, or financial texts—fine-tuned models like **BioBART** or **Longformer-based summarizers** often provide more accurate and context-aware results, particularly when working with long or technical documents.  

#### Recommended Models for Summarization Tasks  

- **Extractive Summarization:** Models like [`sshleifer/distilbart-cnn-12-6`](https://sparknlp.org/2025/02/05/distilbart_cnn_12_6_sshleifer_en.html){:target="_blank"} and [`bertsumext`](https://github.com/nlpyang/PreSumm){:target="_blank"} are effective for selecting the most important sentences directly from the source text.  

- **Abstractive Summarization:** Encoder–decoder models such as [`bart-large-cnn`](https://sparknlp.org/2025/01/26/bart_large_cnn_facebook_en.html){:target="_blank"} and [`t5-base`](https://sparknlp.org/2021/01/08/t5_base_en.html){:target="_blank"} are strong general-purpose choices for generating fluent, rephrased summaries.  

- **Domain-Specific Summarization:** Specialized variants like [`biobart`](https://sparknlp.org/2025/01/24/biobart_base_en.html){:target="_blank"} for biomedical literature or fine-tuned **Longformer-based summarizers** for legal and financial texts provide stronger results in technical or domain-focused contexts.  

Explore the available summarization models at [Spark NLP Models](https://sparknlp.org/models) to find the one that best suits your summarization needs.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import BartTransformer
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

seq2seq = BartTransformer.pretrained("distilbart_cnn_12_6_sshleifer", "en") \
    .setInputCols(["document"]) \
    .setOutputCol("generation") \
    .setTask("summarize:") \
    .setMaxOutputLength(200) \

pipeline = Pipeline(stages=[
    documentAssembler, 
    seq2seq
])

passage = """
Artificial intelligence is transforming industries around the world. 
Healthcare systems are adopting AI to analyze medical images, predict patient outcomes, 
and accelerate the discovery of new drugs. In finance, machine learning algorithms are 
used to detect fraudulent transactions and provide personalized investment advice. 
Transportation is also being reshaped by autonomous vehicles and smarter traffic 
management systems. Despite these benefits, concerns remain about job displacement, 
data privacy, and the ethical use of AI technologies. Governments and organizations 
are working together to create guidelines and regulations that ensure the responsible 
development of AI, while still fostering innovation and economic growth.
"""

data = spark.createDataFrame([[passage]]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.select("generation.result").show(truncate=False)

```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val seq2seq = BartTransformer
  .pretrained("distilbart_cnn_12_6_sshleifer", "en")
  .setInputCols(Array("document"))
  .setOutputCol("generation")
  .setTask("summarize:")
  .setMaxOutputLength(200)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  seq2seq
))

val passage =
  """
  Artificial intelligence is transforming industries around the world.
  Healthcare systems are adopting AI to analyze medical images, predict patient outcomes,
  and accelerate the discovery of new drugs. In finance, machine learning algorithms are
  used to detect fraudulent transactions and provide personalized investment advice.
  Transportation is also being reshaped by autonomous vehicles and smarter traffic
  management systems. Despite these benefits, concerns remain about job displacement,
  data privacy, and the ethical use of AI technologies. Governments and organizations
  are working together to create guidelines and regulations that ensure the responsible
  development of AI, while still fostering innovation and economic growth.
  """

val data = Seq(passage).toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("generation.result").show(false)

```
</div>

<div class="tabs-box" markdown="1">
```
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                                                                                                                                                                                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[ Artificial intelligence is transforming industries around the world . Healthcare systems are adopting AI to analyze medical images and predict patient outcomes . In finance, machine learning algorithms are used to detect fraudulent transactions and provide personalized investment advice . Transportation is also being reshaped by autonomous vehicles and smarter traffic management systems .]|
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of text classification models in real time, visit our interactive demos:

- **[Text summarization](https://demo.johnsnowlabs.com/public/TEXT_SUMMARIZATION/){:target="_blank"}**
- **[Sparknlp Text Summarization](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-bert-annotators){:target="_blank"}**

## Useful Resources

Here are some resources to get you started with summarization in Spark NLP:

**Articles and Guides**
- *[Empowering NLP with Spark NLP and T5 Model: Text Summarization and Question Answering](https://www.johnsnowlabs.com/empowering-nlp-with-spark-nlp-and-t5-model-text-summarization-and-question-answering/){:target="_blank"}*

**Notebooks**
- **Document Summarization with BART** *[1](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/08.Summarization_with_BART.ipynb){:target="_blank"}*, *[2](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/08.Summarization_with_BART.ipynb){:target="_blank"}* 
- *[T5 Workshop with Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/10.1_T5_Workshop_with_Spark_NLP.ipynb){:target="_blank"}*