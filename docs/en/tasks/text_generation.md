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

**Text Generation** is a natural language processing task where models create new text based on a given input. Instead of assigning labels, these models expand, complete, or rephrase text in a coherent way. For example, given *“Once upon a time,”* a text generation model might continue with *“we knew that our ancestors were on the verge of extinction...”*. This task includes both **completion models**, which predict the next word in a sequence to build longer passages, and **text-to-text models**, which map one piece of text to another for tasks like *translation*, *summarization*, or *classification*.

Depending on how they are trained, text generation models come in different variants: **base models** (e.g., Mistral 7B, Llama 3 70B) suited for fine-tuning, **instruction-tuned models** (e.g., Qwen 2, Yi 1.5, Llama 70B Instruct) that follow prompts like *“Write a recipe for chocolate cake”*, and **human feedback models**, which use RLHF to align outputs with human preferences. These capabilities make text generation useful for a wide range of applications, from **chatbots and creative writing** to **code generation and summarization**, with larger models typically producing more fluent and context-aware outputs.

## Picking a Model  

When picking a model for text generation, start by clarifying your goal—whether you need completions, rephrasings, translations, summaries, or creative writing. Base models like **Mistral 7B** or **Llama 3 70B** are good for fine-tuning, while instruction-tuned ones such as **Qwen 2** or **Llama 70B Instruct** work better out of the box for prompts like “Write a recipe for chocolate cake.” Human-feedback models trained with RLHF usually give the most user-aligned responses. For quick or lightweight tasks, smaller models are efficient, while larger ones generally produce more fluent, context-aware text suited for chatbots, code generation, and long-form writing. For specific tasks, **Pegasus**, **BART**, or **Llama 3 Instruct** are strong for summarization; **MarianMT**, **M2M-100**, or **NLLB** excel at translation; **GPT-based models**, **Llama 3 70B Instruct**, and **Yi 1.5** are strong for creative writing; **Code Llama**, **StarCoder**, and **GPT-4 Turbo (Code)** are well-suited for code generation; and for dialogue, **Llama 3 Instruct**, **Qwen 2**, and **GPT-4** provide reliable conversational performance.  

Explore models tailored for text generation at [Spark NLP Models](https://sparknlp.org/models)

<!-- 
#### Recommended Models for Specific Text Generation Tasks

- **Summarization:** Use models like [`t5-base`](https://sparknlp.org/2021/01/08/t5_base_en.html){:target="_blank"} and [`bart-large-cnn`](https://sparknlp.org/2023/05/11/bart_large_cnn_en.html){:target="_blank"} for general-purpose text summarization tasks.
- **Machine Translation:** Consider models such as [`t5_base`](https://sparknlp.org/2021/01/08/t5_base_en.html){:target="_blank"} and [`m2m100_418M`](https://sparknlp.org/2024/05/19/m2m100_418M_xx.html){:target="_blank"} you can also consider searching models with the [`Marian Transformer`](https://sparknlp.org/models?annotator=MarianTransformer){:target="_blank"} Annotator class for translating between non-english languages.
- **Conversational Agents:** For building chatbots and dialogue systems, use models like [`gpt2`](https://sparknlp.org/2021/12/03/gpt2_en.html){:target="_blank"} to generate coherent and contextually aware responses. -->

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import AutoGGUFModel
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

auto_gguf_model = AutoGGUFModel.pretrained("qwen3_4b_q4_k_m_gguf", "en") \
    .setInputCols(["document"]) \
    .setOutputCol("completions") \
    .setBatchSize(4) \
    .setNPredict(20) \
    .setNGpuLayers(99) \
    .setTemperature(0.4) \
    .setTopK(40) \
    .setTopP(0.9) \
    .setPenalizeNl(True)

pipeline = Pipeline().setStages([
    document_assembler,
    auto_gguf_model
])

data = spark.createDataFrame([
    ["A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?"]
]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val autoGGUFModel = AutoGGUFModel.pretrained("qwen3_4b_q4_k_m_gguf", "en")
  .setInputCols("document")
  .setOutputCol("completions")
  .setBatchSize(4)
  .setNPredict(20)
  .setNGpuLayers(99)
  .setTemperature(0.4f)
  .setTopK(40)
  .setTopP(0.9f)
  .setPenalizeNl(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  autoGGUFModel
))

val data = Seq("A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?").toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

```
</div>

<div class="tabs-box" markdown="1">
```
Explanation:
The phrase "all but 9 run away" means that 9 sheep did not run away, while the remaining (17 - 9 = 8) did. Therefore, the farmer still has the 9 sheep that stayed behind.
Answer: 9.
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of text generation models in real time, visit our interactive demos:

- **[Generative Pre-trained Transformer 2 (OpenAI GPT2)](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-gpt2){:target="_blank"}**
- **[Text-To-Text Transfer Transformer (Google T5)](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-t5){:target="_blank"}**
- **[SQL Query Generation](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-text-to-sql-t5){:target="_blank"}**
- **[Multilingual Text Translation with MarianMT](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-MarianMT){:target="_blank"}**

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
