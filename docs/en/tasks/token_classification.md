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

**Token classification** is a natural language understanding task where labels are assigned to individual tokens in a text. Common subtasks include **Named Entity Recognition (NER)** and **Part-of-Speech (PoS)** tagging. For example, NER models can be trained to detect entities like dates, people, and locations, while PoS tagging identifies whether a word functions as a noun, verb, punctuation mark, or another grammatical category.

<div style="text-align: center;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/B3xB9gaBosw?si=hDgXLUoduQkkodPN&amp;start=258" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

## Picking a Model

When picking a model for token classification, start with the type of task you need—such as **Named Entity Recognition (NER)** for tagging names of people, places, or organizations, **Part-of-Speech (POS) tagging** for grammatical structure, or **slot filling** in chatbots. For small or less complex datasets, lighter models like **DistilBERT** or **pretrained pipelines** can give fast and practical results. If you have more data or need higher accuracy, larger models like **BERT**, **RoBERTa**, or **XLM-R** are strong baselines, and domain-specialized versions like **BioBERT** (for biomedical text) or **Legal-BERT** (for legal text) often perform best in their fields. Keep in mind trade-offs: smaller models are faster and easier to deploy, while larger transformers provide richer context understanding but come with higher compute costs.

You can explore and select models for your token classification tasks at [Spark NLP Models](https://sparknlp.org/models)

#### Recommended Models for Specific Token Classification Tasks

- **Named Entity Recognition (NER):** Use models like [`bert-base-NER`](https://sparknlp.org/2022/05/09/bert_ner_bert_base_NER_en_3_0.html){:target="_blank"} and [`xlm-roberta-large-finetuned-conll03-english`](https://sparknlp.org/2022/08/14/xlmroberta_ner_large_finetuned_conll03_english_xx_3_0.html){:target="_blank"} for general-purpose NER tasks.
- **Part-of-Speech Tagging (POS):** For POS tagging, consider using models such as [`pos_anc`](https://sparknlp.org/2021/03/05/pos_anc.html){:target="_blank"}.
- **Healthcare NER:** For clinical texts, [`ner_jsl`](https://nlp.johnsnowlabs.com/2022/10/19/ner_jsl_en.html){:target="_blank"} and [`pos_clinical`](https://sparknlp.org/2023/02/17/ner_jsl_en.html){:target="_blank"} is tailored for extracting medical entities.

If existing models do not meet your requirements, you can train your own custom model using the [Spark NLP Training Documentation](https://sparknlp.org/docs/en/training).

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

tokenClassifier = BertForTokenClassification.pretrained() \
    .setInputCols(["token", "document"]) \
    .setOutputCol("label") \
    .setCaseSensitive(True)

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    tokenClassifier
])

data = spark.createDataFrame([["John Lenon was born in London and lived in Paris. My name is Sarah and I live in London"]]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.select("label.result").show(truncate=False)

```

```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  tokenClassifier
))

val data = Seq("John Lenon was born in London and lived in Paris. My name is Sarah and I live in London").toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("label.result").show(false)

```
</div>

<div class="tabs-box" markdown="1">
```
+------------------------------------------------------------------------------------+
|result                                                                              |
+------------------------------------------------------------------------------------+
|[B-PER, I-PER, O, O, O, B-LOC, O, O, O, B-LOC, O, O, O, O, B-PER, O, O, O, O, B-LOC]|
+------------------------------------------------------------------------------------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of text classification models in real time, visit our interactive demos:

- **[Recognize Entities - Live Demos & Notebooks](https://sparknlp.org/recognize_entitie){:target="_blank"}**
- **[BERT Annotators Demo](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-bert-annotators){:target="_blank"}**
- **[Named Entity Recognition (NER)](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-named-entity-recognition){:target="_blank"}**
- **[POS Tagging](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-POS-tagging){:target="_blank"}**

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