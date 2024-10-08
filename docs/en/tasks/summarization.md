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

**Summarization** is the task of generating concise and informative summaries from longer documents. This is useful for a wide range of applications, such as summarizing news articles, legal documents, or any large texts where key points need to be extracted. Spark NLP offers advanced summarization models that can create high-quality summaries efficiently.

Summarization models take input text and generate shorter versions while preserving essential information. Common use cases include:

- **News Summaries:** Automatically condensing long news articles into brief, digestible summaries.
- **Legal Documents:** Summarizing lengthy contracts, case studies, or legal opinions.
- **Research Papers:** Extracting key insights and conclusions from scientific papers.

By leveraging summarization models, organizations can efficiently process large amounts of textual data and extract critical information, making it easier to consume and understand complex documents.

## Picking a Model

When choosing a summarization model, consider factors like the **length of the input text** and the **desired summary style** (e.g., extractive or abstractive). Some models are better suited for shorter inputs, while others excel in handling long documents. Evaluate whether your task requires **sentence-level summaries** or **paragraph-level condensation**.

Consider the **domain** of the text, such as legal, scientific, or general news, as domain-specific models often perform better. Explore the available summarization models at [Spark NLP Models](https://sparknlp.org/models) to find the one that best suits your summarization needs.

#### Recommended Models for Summarization Tasks

- **General Summarization:** For most summarization tasks, consider models like [`bart-large-cnn`](https://sparknlp.org/2023/05/11/bart_large_cnn_en.html){:target="_blank"} and [`t5-base`](https://sparknlp.org/2021/01/08/t5_base_en.html){:target="_blank"} are well suited for generating concise summaries.

By selecting the right model, you can efficiently condense long documents into meaningful summaries, saving time and effort.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# Step 1: Assemble raw text data into a format that Spark NLP can process
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

# Step 2: Load a pretrained BART model for summarization
bart = BartTransformer.pretrained("distilbart_xsum_12_6") \
    .setTask("summarize:") \
    .setInputCols(["documents"]) \
    .setMaxOutputLength(200) \
    .setOutputCol("summaries")

# Step 3: Create a pipeline with the document assembler and BART model
pipeline = Pipeline().setStages([documentAssembler, bart])

# Step 4: Sample data - a long text passage for summarization
data = spark.createDataFrame([[
    "Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a " +
    "downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness" +
    " of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this " +
    "paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework " +
    "that converts all text-based language problems into a text-to-text format. Our systematic study compares " +
    "pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens " +
    "of language understanding tasks. By combining the insights from our exploration with scale and our new " +
    "Colossal Clean Crawled Corpus, we achieve state-of-the-art results on many benchmarks covering " +
    "summarization, question answering, text classification, and more. To facilitate future work on transfer " +
    "learning for NLP, we release our data set, pre-trained models, and code."
]]).toDF("text")

# Step 5: Apply the pipeline to generate the summary
result = pipeline.fit(data).transform(data)

# Step 6: Display the summary
result.select("summaries.result").show(truncate=False)

# +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |result                                                                                                                                                                          |
# +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |[transfer learning has emerged as a powerful technique in natural language processing (NLP) the effectiveness of transfer learning has given rise to a diversity of approaches, |
# |methodologies, and practice .]                                                                                                                                                  |
# +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.seq2seq.GPT2Transformer
import org.apache.spark.ml.Pipeline

// Step 1: Document Assembler to prepare the text data
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

// Step 2: Load BART model for text generation with customization
val bart = BartTransformer.pretrained("distilbart_xsum_12_6")
  .setInputCols(Array("documents"))
  .setMinOutputLength(10)
  .setMaxOutputLength(30)
  .setDoSample(true)
  .setTopK(50)
  .setOutputCol("generation")

// Step 3: Define the pipeline stages
val pipeline = new Pipeline().setStages(Array(documentAssembler, bart))

// Step 4: Input text data to be summarized
val data = Seq(
  "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
  "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
  "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
).toDF("text")

// Step 5: Fit the model and apply the pipeline
val result = pipeline.fit(data).transform(data)

// Step 6: Show the generated summary
results.select("generation.result").show(truncate = false)

// +--------------------------------------------------------------+
// |result                                                        |
// +--------------------------------------------------------------+
// |[Nearly 800 thousand customers were affected by the shutoffs.]|
// +--------------------------------------------------------------+

```
</div>

## Try Real-Time Demos!

If you want to see the outputs of text classification models in real time, visit our interactive demos:

- **[Sparknlp Text Summarization](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-bert-annotators){:target="_blank"}** – A live demo where you can try your inputs on text classification models on the go.
- **[Text summarization](https://demo.johnsnowlabs.com/public/TEXT_SUMMARIZATION/){:target="_blank"}** – An interactive demo for sentiment and emotion detection.

## Useful Resources

Here are some resources to get you started with summarization in Spark NLP:

**Articles and Guides**
- *[Empowering NLP with Spark NLP and T5 Model: Text Summarization and Question Answering](https://www.johnsnowlabs.com/empowering-nlp-with-spark-nlp-and-t5-model-text-summarization-and-question-answering/){:target="_blank"}*

**Notebooks**
- **Document Summarization with BART** *[1](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/08.Summarization_with_BART.ipynb){:target="_blank"}*, *[2](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/08.Summarization_with_BART.ipynb){:target="_blank"}* 
- *[T5 Workshop with Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/10.1_T5_Workshop_with_Spark_NLP.ipynb){:target="_blank"}*