---
layout: docs
header: true
seotitle:
title: Zero-Shot Classification
permalink: docs/en/tasks/zero_shot_classification
key: docs-tasks-zero_shot_classification
modify_date: "2024-09-26"
show_nav: true
sidebar:
  nav: sparknlp
---

**Zero-Shot Text Classification** is a natural language processing task where a model can assign text to *categories it was never explicitly trained on*. Instead of relying only on predefined classes, it uses general language understanding to match new inputs to a list of **candidate labels** given at inference time. For example, given the text *“Dune is the best movie ever”* and labels *CINEMA, ART, MUSIC*, a zero-shot model might output *CINEMA (0.900)*, *ART (0.100)*, and *MUSIC (0.000)*. This approach is especially useful when labeled data is limited or when new categories appear after training.

Zero-shot classification works by pairing a **natural language prompt** with candidate labels, without needing any examples of the task itself. This differs from single-shot or few-shot classification, which provide one or a few examples of the task. The ability to perform zero-, single-, and few-shot tasks emerges in **large language models** (typically with 100M+ parameters), with performance improving as models scale. Thanks to this, zero-shot classification is widely used for tasks like **sentiment analysis, topic labeling, and intent detection**, making it a flexible solution for real-world applications without retraining.

## Picking a Model

When picking a model for zero-shot classification, first check if a task-specific model already exists—because trained models usually outperform zero-shot ones. Smaller models like **DistilBART-MNLI** are fast and efficient, while larger ones such as **RoBERTa-MNLI** or **GPT-based classifiers** deliver higher accuracy and handle nuance better. In specialized domains, domain-adapted or prompt-tuned models are usually worth exploring.

#### Recommended Models for Zero-Shot Classification

- **Zero-Shot Text Classification:** Consider using models like [`bart-large-mnli`](https://sparknlp.org/2024/08/27/bart_large_zero_shot_classifier_mnli_en.html){:target="_blank"} for general-purpose multilingual text data classification across various domains.
- **Zero-Shot Named Entity Recognition (NER):** Use models like [`zero_shot_ner_roberta`](https://sparknlp.org/2023/02/08/zero_shot_ner_roberta_en.html){:target="_blank"} for identifying entities across various domains and languages without requiring task-specific labeled data.

You can explore and select models for your zero-shot classification tasks at [Spark NLP Models](https://sparknlp.org/models)

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

sequenceClassifier = BertForZeroShotClassification.pretrained() \
    .setInputCols(["token", "document"]) \
    .setOutputCol("label") \
    .setCandidateLabels(["technology", "health", "finance"])

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    sequenceClassifier
])

data = spark.createDataFrame([
  ("The new iPhone release has sparked debates about innovation in technology.",),
  ("Doctors recommend regular exercise to maintain good health.",),
  ("The stock market experienced strong gains this week as investor confidence boosted financial markets.",)
], ["text"])

model = pipeline.fit(data)
result = model.transform(data)

result.select("text", "label.result").show(truncate=False)

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
  .setOutputCol("token")

val sequenceClassifier = BertForZeroShotClassification.pretrained()
  .setInputCols(Array("token", "document"))
  .setOutputCol("label")
  .setCandidateLabels(Array("technology", "health", "finance"))

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  sequenceClassifier
))

val data = Seq(
  ("The new iPhone release has sparked debates about innovation in technology."),
  ("Doctors recommend regular exercise to maintain good health."),
  ("The stock market experienced strong gains this week as investor confidence boosted financial markets.")
).toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("text", "label.result").show(false)

```
</div>

<div class="tabs-box" markdown="1">
```
+-----------------------------------------------------------------------------------------------------+------------+
|text                                                                                                 |result      |
+-----------------------------------------------------------------------------------------------------+------------+
|The new iPhone release has sparked debates about innovation in technology.                           |[technology]|
|Doctors recommend regular exercise to maintain good health.                                          |[health]    |
|The stock market experienced strong gains this week as investor confidence boosted financial markets.|[finance]   |
+-----------------------------------------------------------------------------------------------------+------------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of text classification models in real time, visit our interactive demos:

- **[BERT Annotators Demo](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-bert-annotators){:target="_blank"}**
- **[Zero-Shot Named Entity Recognition (NER)](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-Zero-Shot-NER){:target="_blank"}**

## Useful Resources

**Notebooks**
- *[Zero-Shot Text Classification in Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.4_ZeroShot_Text_Classification.ipynb){:target="_blank"}*
- *[Zero-Shot for Named Entity Recognition](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/4.2_ZeroShot_NER.ipynb){:target="_blank"}*
