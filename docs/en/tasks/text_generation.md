---
layout: docs  
header: true  
seotitle:  
title: Text Generation  
permalink: docs/en/tasks/text_generation  
key: docs-tasks-text-generation  
modify_date: "2024-09-28"  
show_nav: true  
sidebar:  
  nav: sparknlp  
---

**Text generation** is the task of generating meaningful text based on a given input. It is widely used in various *natural language processing (NLP)* applications such as summarization, machine translation, conversational agents, and more. Spark NLP provides SOTA solutions for text generation, enabling you to produce high-quality and contextually relevant text outputs.

Text generation models create text sequences by predicting the next word or sequence of words based on the input prompt. Common use cases include:

- **Summarization:** Automatically generating concise summaries from longer text.
- **Machine Translation:** Translating text from one language to another while maintaining meaning and fluency.
- **Conversational Agents:** Building intelligent systems that can hold natural and coherent conversations with users.

By leveraging text generation, organizations can build systems capable of generating human-like text, making it useful for content creation, automated writing, and more.

<!-- <div style="text-align: center;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/dQw4w9WgXcQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div> -->

## Picking a Model

When selecting a model for text generation, consider several important factors. First, determine the **type of output** you require (e.g., summarization, translation, or free-form generation). Decide whether your task needs **structured output** like summaries or **creative text generation**.

Next, evaluate the **style and language** of the data you'll be working with—are you dealing with formal language (e.g., research papers) or informal language (e.g., social media)? Model performance metrics such as **perplexity**, **BLEU score**, or **ROUGE score** are also crucial for understanding the quality of the generated text. Finally, take into account the **computational resources** available, as some models (e.g., GPT or T5) may require significant memory and processing power.

Explore models tailored for text generation at [Spark NLP Models](https://sparknlp.org/models), where you’ll find various options for different text generation tasks.

#### Recommended Models for Specific Text Generation Tasks

- **Summarization:** Use models like [`t5-base`](https://sparknlp.org/2021/01/08/t5_base_en.html){:target="_blank"} and [`bart-large-cnn`](https://sparknlp.org/2023/05/11/bart_large_cnn_en.html){:target="_blank"} for general-purpose text summarization tasks.
- **Machine Translation:** Consider models such as [`t5_base`](https://sparknlp.org/2021/01/08/t5_base_en.html){:target="_blank"} and [`m2m100_418M`](https://sparknlp.org/2024/05/19/m2m100_418M_xx.html){:target="_blank"} you can also consider searching models with the [`Marian Transformer`](https://sparknlp.org/models?annotator=MarianTransformer){:target="_blank"} Annotator class for translating between non-english languages.
- **Conversational Agents:** For building chatbots and dialogue systems, use models like [`gpt2`](https://sparknlp.org/2021/12/03/gpt2_en.html){:target="_blank"} to generate coherent and contextually aware responses.

By selecting the appropriate text generation model, you can enhance your ability to produce contextually rich and meaningful text outputs tailored to your specific NLP tasks.
  
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
    .setOutputCol("documents")

# Loading a pre-trained text generation model
# You can replace `T5Transformer.pretrained("t5_small", "xx")` with your selected model and the transformer it's based on
# For example: BartTransformer.pretrained("bart_large_cnn")
t5 = T5Transformer.pretrained("t5_small", "xx") \
    .setTask("summarize:") \
    .setInputCols(["documents"]) \
    .setMaxOutputLength(200) \
    .setOutputCol("summaries")

# Defining the pipeline with document assembler, tokenizer, and classifier
pipeline = Pipeline().setStages([documentAssembler, t5])

# Creating a sample DataFrame
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

# Fitting the pipeline and transforming the data
result = pipeline.fit(data).transform(data)

# Showing the results
result.select("summaries.result").show(truncate=False)

+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                          |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[transfer learning has emerged as a powerful technique in natural language processing (NLP) the effectiveness of transfer learning has given rise to a diversity of approaches, |
| methodologies, and practice .]                                                                                                                                                 |
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+ 
```

```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer
import org.apache.spark.ml.Pipeline

// Step 1: Assembling the document from the input text
// Converts the input 'text' column into a 'document' column, required for NLP tasks
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

// Step 3: Loading a pre-trained BERT model for token classification
// Applies a pre-trained BERT model for Named Entity Recognition (NER) to classify tokens
// `T5Transformer.pretrained()` loads the model, and `setInputCols` defines the input columns
val t5 = T5Transformer.pretrained("t5_small")
  .setTask("summarize:")
  .setInputCols(Array("documents"))
  .setMaxOutputLength(200)
  .setOutputCol("summaries")

// Step 4: Defining the pipeline
// The pipeline stages are document assembler, tokenizer, and token classifier
val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

// Step 5: Creating a sample DataFrame
// Creates a DataFrame with a sample sentence that will be processed by the pipeline
val data = Seq(
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
).toDF("text")

// Step 6: Fitting the pipeline and transforming the data
// The pipeline is fitted on the input data, then it performs the transformation to generate token labels
val result = pipeline.fit(data).transform(data)

// Step 7: Showing the results
// Displays the 'label.result' column, which contains the Named Entity Recognition (NER) labels for each token
result.select("summaries.result").show(false)

+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                          |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[transfer learning has emerged as a powerful technique in natural language processing (NLP) the effectiveness of transfer learning has given rise to a diversity of approaches, |
|methodologies, and practice .]                                                                                                                                                  |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of text generation models in real time, visit our interactive demos:

- **[Generative Pre-trained Transformer 2 (OpenAI GPT2)](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-gpt2){:target="_blank"}** – GPT-2 generates human-like text from prompts.
- **[Text-To-Text Transfer Transformer (Google T5)](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-t5){:target="_blank"}** – T5 performs text tasks like summarization and translation.
- **[SQL Query Generation](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-text-to-sql-t5){:target="_blank"}** – Converts natural language commands into SQL queries.
- **[Multilingual Text Translation with MarianMT](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-MarianMT){:target="_blank"}** – Translates text between multiple languages.

## Useful Resources

Want to dive deeper into text generation with Spark NLP? Here are some curated resources to help you get started and explore further:

**Articles and Guides**
- *[Empowering NLP with Spark NLP and T5 Model: Text Summarization and Question Answering](https://www.johnsnowlabs.com/empowering-nlp-with-spark-nlp-and-t5-model-text-summarization-and-question-answering/){:target="_blank"}*
- *[Multilingual machine translation with Spark NLP](https://www.johnsnowlabs.com/multilingual-machine-translation-with-spark-nlp/){:target="_blank"}*

**Notebooks** 
- *[GPT2Transformer: OpenAI Text-To-Text Transformer](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/14.GPT2_Transformer_In_Spark_NLP.ipynb){:target="_blank"}*
- *[LLAMA2Transformer: CausalLM wiht Open Source models](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/22.0_Llama2_Transformer_In_SparkNLP.ipynb){:target="_blank"}*
- *[SQL Code Generation and Style Transfer with T5](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/10.2_SQL_Code_Generation_and_Style_Transfer_with_T5.ipynb){:target="_blank"}*
- *[T5 Workshop with Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/10.1_T5_Workshop_with_Spark_NLP.ipynb){:target="_blank"}*
- *[Translation in Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/20.0_Translations.ipynb){:target="_blank"}*
- *[Summarization in Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/10.2_SQL_Code_Generation_and_Style_Transfer_with_T5.ipynb){:target="_blank"}*
- *[OpenAI in SparkNLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/22.1_OpenAI_In_SparkNLP.ipynb){:target="_blank"}*
