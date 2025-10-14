--- 
layout: docs  
header: true  
seotitle:  
title: Translation  
permalink: docs/en/tasks/translation  
key: docs-tasks-translation  
modify_date: "2024-09-28"  
show_nav: true  
sidebar:  
  nav: sparknlp  
---

**Translation** is a natural language processing task where models convert text from one language into another while preserving its meaning, grammar, and context. For example, given the input *“My name is Omar and I live in Zürich”*, a translation model might output *“Mein Name ist Omar und ich wohne in Zürich”*. Modern translation models, especially **multilingual neural models** like **mBART**, can handle a wide variety of language pairs and can also be fine-tuned on custom data to improve accuracy for specific domains or dialects.

Translation models are widely used to build **multilingual conversational agents** and cross-lingual applications. They can either translate datasets of user intents and responses to train a new model in the target language or translate live user inputs and chatbot outputs for real-time interaction. These capabilities make translation essential for **global communication, content localization, cross-border business, and international customer support**, enabling systems to operate seamlessly across multiple languages.

## Picking a Model  

The choice of model for translation depends on the languages, domain, and whether real-time or batch translation is required. For **general-purpose multilingual translation**, encoder–decoder architectures like **mBART**, **M2M100**, and **MarianMT** perform well across a wide range of language pairs. For **high-quality domain-specific translation**, fine-tuned versions of these models can be used, such as models trained on legal, medical, or technical corpora. **Lightweight or faster models** like **DistilMarianMT** or distilled versions of **mBART** are suitable for real-time applications or deployment in resource-constrained environments. Finally, when **rare or low-resource languages** are involved, models like **NLLB-200** or language-adapted versions of **M2M100** provide improved coverage and accuracy.

#### Recommended Models for Translation Tasks

- **General-Purpose Multilingual Translation:** Models such as [`mbart-large-50-many-to-many-mmt`](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt){:target="_blank"}, [`m2m100_418M`](https://sparknlp.org/2024/05/19/m2m100_418M_xx.html){:target="_blank"}, and [`Helsinki-NLP/opus-mt`](https://sparknlp.org/models?q=Helsinki-NLP%2Fopus-mt&type=model&sort=downloads&annotator=MarianTransformer){:target="_blank"} handle a wide variety of language pairs effectively.  

- **Domain-Specific Translation:** For legal, medical, technical, or other specialized texts, fine-tuned variants of **mBART**, **M2M100**, or **MarianMT** trained on domain-specific corpora provide higher accuracy.  

- **Lightweight or Real-Time Translation:** Distilled or smaller models like [`Helsinki-NLP/opus-mt`](https://sparknlp.org/models?q=Helsinki-NLP%2Fopus-mt&type=model&sort=downloads&annotator=MarianTransformer){:target="_blank"} and distilled **mBART** versions are optimized for low-latency, resource-constrained deployment.  

- **Low-Resource Languages:** Models such as [`NLLB-200`](https://sparknlp.org/2024/11/27/nllb_distilled_600M_8int_xx.html){:target="_blank"} or language-adapted versions of **M2M100** are recommended for improved performance on rare or low-resource language pairs.  

Explore the available translation models at [Spark NLP Models](https://sparknlp.org/models) to find the one that best suits your translation tasks.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

m2m100 = M2M100Transformer.pretrained("m2m100_418M") \
    .setInputCols(["documents"]) \
    .setMaxOutputLength(50) \
    .setOutputCol("generation") \
    .setSrcLang("zh") \   # Source language: Chinese
    .setTgtLang("en")     # Target language: English

pipeline = Pipeline().setStages([
  documentAssembler, 
  m2m100
])

data = spark.createDataFrame([["生活就像一盒巧克力。"]]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.select("summaries.generation").show(truncate=False)

```

```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

val m2m100 = M2M100Transformer.pretrained("m2m100_418M")
  .setInputCols(Array("documents"))
  .setSrcLang("zh")          // Source language: Chinese
  .serTgtLang("en")         // Target language: English
  .setMaxOutputLength(100)
  .setDoSample(false)        
  .setOutputCol("generation")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler, 
  m2m100
))

val data = Seq("生活就像一盒巧克力。").toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("generation.result").show(truncate = false)

```
</div>

<div class="tabs-box" markdown="1">
```
+-------------------------------------------------------------------------------------------+
|result                                                                                     |
+-------------------------------------------------------------------------------------------+
|[ Life is like a box of chocolate.]                                                        |
+-------------------------------------------------------------------------------------------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of text generation models in real time, visit our interactive demos:

- **[Text-To-Text Transfer Transformer (Google T5)](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-t5){:target="_blank"}**
- **[Multilingual Text Translation with MarianMT](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-MarianMT){:target="_blank"}**
- **[M2M100 Multilingual Translation Model](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-M2M100){:target="_blank"}**

## Useful Resources

Want to dive deeper into text generation with Spark NLP? Here are some curated resources to help you get started and explore further:

**Articles and Guides**
- *[Multilingual machine translation with Spark NLP](https://www.johnsnowlabs.com/multilingual-machine-translation-with-spark-nlp/){:target="_blank"}*
- *[Use Spark NLP offline models for Language Translation](https://www.linkedin.com/pulse/use-spark-nlp-offline-models-language-translation-mei-wu/){:target="_blank"}*

**Notebooks** 
- *[T5 Workshop with Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/10.1_T5_Workshop_with_Spark_NLP.ipynb){:target="_blank"}*
- *[Translation in Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/20.0_Translations.ipynb){:target="_blank"}*