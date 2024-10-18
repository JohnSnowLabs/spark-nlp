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

**Translation** is the task of converting text from one language into another. This is essential for multilingual applications such as content localization, cross-language communication, and more. Spark NLP offers advanced translation models that provide high-quality translations between multiple languages.

Translation models process input text in the source language and generate a corresponding translation in the target language. Common use cases include:

- **Cross-Language Communication:** Enabling communication across different languages for global teams.
- **Document Translation:** Translating long-form content such as reports, articles, or manuals.

By using Spark NLP translation models, you can build scalable translation systems to meet your multilingual needs efficiently and accurately.

## Picking a Model

When choosing a translation model, consider factors such as the **source and target languages** and the **size of the input text**. Some models may specialize in specific language pairs or offer better performance for certain types of text (e.g., formal versus informal content). Evaluate whether you need **document-level translation** or **sentence-level translation** based on the use case.

Explore the available translation models at [Spark NLP Models](https://sparknlp.org/models) to find the one that best suits your translation tasks.

#### Recommended Models for Translation Tasks

- **General Translation:** Consider models such as [`t5_base`](https://sparknlp.org/2021/01/08/t5_base_en.html){:target="_blank"} and [`m2m100_418M`](https://sparknlp.org/2024/05/19/m2m100_418M_xx.html){:target="_blank"} you can also consider searching models with the [`Marian Transformer`](https://sparknlp.org/models?annotator=MarianTransformer){:target="_blank"} Annotator class for translating between non-english languages.

Selecting the appropriate model will ensure you produce accurate and fluent translations, tailored to your specific language pair and domain.

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
    .setOutputCol("documents")

# M2M100 Transformer: Loads the pretrained translation model for English to French
m2m100 = M2M100Transformer.pretrained("m2m100_418M") \
    .setInputCols(["documents"]) \
    .setMaxOutputLength(50) \
    .setOutputCol("generation") \
    .setSrcLang("zh") \   # Source language: Chinese
    .setTgtLang("en")     # Target language: English

# Pipeline: Assembles the document assembler and the M2M100 translation model
pipeline = Pipeline().setStages([documentAssembler, m2m100])

# Input Data: A small example dataset is created and converted to a DataFrame
data = spark.createDataFrame([["生活就像一盒巧克力。"]]).toDF("text")

# Running the Pipeline: Fits the pipeline to the data and generates translations
result = pipeline.fit(data).transform(data)

# Output: Displays the translated result
result.select("summaries.generation").show(truncate=False)

+-------------------------------------------------------------------------------------------+
|result                                                                                     |
+-------------------------------------------------------------------------------------------+
|[ Life is like a box of chocolate.]                                                        |
+-------------------------------------------------------------------------------------------+
```

```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.seq2seq.M2M100Transformer
import org.apache.spark.ml.Pipeline

// Document Assembler: Converts input text into a suitable format for NLP processing
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

// M2M100 Transformer: Loads the pretrained translation model for Chinese to English
val m2m100 = M2M100Transformer.pretrained("m2m100_418M")
  .setInputCols(Array("documents"))
  .setSrcLang("zh")          // Source language: Chinese
  .serTgtLang("en")         // Target language: English
  .setMaxOutputLength(100)
  .setDoSample(false)        
  .setOutputCol("generation")

// Pipeline: Assembles the document assembler and the M2M100 translation model
val pipeline = new Pipeline().setStages(Array(documentAssembler, m2m100))

// Input Data: A small example dataset is created and converted to a DataFrame
val data = Seq("生活就像一盒巧克力。").toDF("text")

// Running the Pipeline: Fits the pipeline to the data and generates translations
val result = pipeline.fit(data).transform(data)

// Output: Displays the translated result
result.select("generation.result").show(truncate = false)

+-------------------------------------------------------------------------------------------+
|result                                                                                     |
+-------------------------------------------------------------------------------------------+
|[ Life is like a box of chocolate.]                                                        |
+-------------------------------------------------------------------------------------------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of text generation models in real time, visit our interactive demos:

- **[Text-To-Text Transfer Transformer (Google T5)](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-t5){:target="_blank"}** – T5 performs text tasks like summarization and translation.
- **[Multilingual Text Translation with MarianMT](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-MarianMT){:target="_blank"}** – Translates text between multiple languages.
- **[M2M100 Multilingual Translation Model](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-M2M100){:target="_blank"}** – Translates text between multiple languages.

## Useful Resources

Want to dive deeper into text generation with Spark NLP? Here are some curated resources to help you get started and explore further:

**Articles and Guides**
- *[Multilingual machine translation with Spark NLP](https://www.johnsnowlabs.com/multilingual-machine-translation-with-spark-nlp/){:target="_blank"}*
- *[Use Spark NLP offline models for Language Translation](https://www.linkedin.com/pulse/use-spark-nlp-offline-models-language-translation-mei-wu/){:target="_blank"}*

**Notebooks** 
- *[T5 Workshop with Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/10.1_T5_Workshop_with_Spark_NLP.ipynb){:target="_blank"}*
- *[Translation in Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/20.0_Translations.ipynb){:target="_blank"}*