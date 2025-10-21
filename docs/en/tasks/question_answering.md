---
layout: docs  
header: true  
seotitle:  
title: Question Answering
permalink: docs/en/tasks/question_answering
key: docs-tasks-question-answering
modify_date: "2024-09-28"  
show_nav: true  
sidebar:  
  nav: sparknlp  
---

**Question Answering (QA)** is a natural language processing task where models provide answers to questions using a given context, or in some cases, from their own knowledge without any context. For example, given the question *"Which name is also used to describe the Amazon rainforest in English?"* and the context *"The Amazon rainforest, also known in English as Amazonia or the Amazon Jungle"*, a QA model would output *"Amazonia"*. This capability makes QA ideal for searching within documents and for powering systems like **FAQ automation**, **customer support**, and **knowledge-base search**.

### Types of Question Answering

There are several QA variants: 
- **Extractive QA**, which pulls the exact answer span from a context and is often solved with BERT-like models
- **Open generative QA**, which generates natural-sounding answers based on a provided context
- **Closed generative QA**, where the model answers entirely from its internal knowledge without any context. 

QA systems can also be **open-domain**, covering a wide range of topics, or **closed-domain**, focused on specialized areas such as law or medicine. Together, these variants make QA a core building block for modern applications such as **search engines**, **chatbots**, and **virtual assistants**.
∂
## Picking a Model

Depending on the QA variant and domain, different models are typically favored: **BERT**, **RoBERTa**, and **ALBERT** are strong choices for extractive QA; **T5**, **BART**, and newer families like **LLaMA 2** excel at open generative QA; models such as **LLaMA 2** and **Mistral** are well suited for closed generative QA; and domain-specific variants like **BioBERT**, **LegalBERT**, and **SciBERT** deliver the highest performance for specialized fields. A good rule of thumb is to use **extractive QA** when precise span-level answers are needed from a document, **open generative QA** when natural and context-aware responses are desired, and **closed generative QA** when relying on a model’s internal knowledge is sufficient or preferable.

### Recommended Models for Specific QA Tasks  

- **Extractive QA:** Models such as [`distilbert-base-cased-distilled-squad`](https://sparknlp.org/2023/11/26/distilbert_base_cased_qa_squad2_en.html){:target="_blank"} and [`bert-large-uncased-whole-word-masking-finetuned-squad`](https://sparknlp.org/2024/09/01/bert_large_uncased_whole_word_masking_finetuned_squad_google_bert_en.html){:target="_blank"} are well suited for identifying precise answer spans directly from context.  

- **Open Generative QA (context-based):** For generating fluent, context-aware answers, models like [`t5_base`](https://sparknlp.org/2021/01/08/t5_base_en.html){:target="_blank"} and [`bart_base`](https://sparknlp.org/2025/02/08/qa_facebook_bart_base_ibrahimgiki_en.html){:target="_blank"} provide strong performance.  

- **Closed Generative QA (knowledge-based):** When answers must be drawn from the model’s internal knowledge rather than external context, options such as [`roberta-base-squad2`](https://sparknlp.org/2022/12/02/roberta_qa_deepset_base_squad2_en.html){:target="_blank"} and newer families like [`llama_2_7b_chat`](https://sparknlp.org/2024/05/19/llama_2_7b_chat_hf_int8_en.html){:target="_blank"} are effective choices.  

- **Domain-Specific QA:** For specialized use cases, consider domain-adapted models such as [`dmis-lab/biobert-large-cased-v1.1-squad`](https://sparknlp.org/2023/11/14/bert_qa_biobert_large_cased_v1.1_squad_en.html){:target="_blank"} for biomedical tasks, [`Beri/legal-qa`](http://127.0.0.1:4000/docs/en/tasks/question_answering){:target="_blank"} for legal texts, or [`ktrapeznikov/scibert_scivocab_uncased_squad_v2`](https://sparknlp.org/2023/11/15/bert_qa_scibert_scivocab_uncased_squad_v2_en.html){:target="_blank"} for scientific literature.  

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = MultiDocumentAssembler() \
    .setInputCols(["question", "context"]) \
    .setOutputCols(["document_question", "document_context"])

spanClassifier = DistilBertForQuestionAnswering.pretrained("distilbert_base_cased_qa_squad2", "en") \
    .setInputCols(["document_question", "document_context"]) \
    .setOutputCol("answer") \
    .setCaseSensitive(True)

pipeline = Pipeline(stages=[
    documentAssembler,
    spanClassifier
])

data = spark.createDataFrame([
    ["What is my name?", "My name is Clara and I live in Berkeley."]
]).toDF("question", "context")

model = pipeline.fit(data)
result = model.transform(data)

result.select("question", "context", "answer.result").show(truncate=False)

```

```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new MultiDocumentAssembler()
  .setInputCols(Array("question", "context"))
  .setOutputCols(Array("document_question", "document_context"))

val spanClassifier = DistilBertForQuestionAnswering.pretrained("distilbert_base_cased_qa_squad2", "en")
  .setInputCols(Array("document_question", "document_context"))
  .setOutputCol("answer")
  .setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  spanClassifier
))

val data = Seq(
  ("What is my name?", "My name is Clara and I live in Berkeley.")
).toDF("question", "context")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("question", "context", "answer.result").show(truncate = false)

```
</div>

<div class="tabs-box" markdown="1">
```
+----------------+----------------------------------------+-------+
|question        |context                                 |result |
+----------------+----------------------------------------+-------+
|What is my name?|My name is Clara and I live in Berkeley.|[Clara]|
+----------------+----------------------------------------+-------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of question answering models in real time, visit our interactive demos:

- **[Multihop QA with BERT](https://sparknlp.org/question_answering){:target="_blank"}**
- **[BERT for Extractive Question Answering](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-bert-qa){:target="_blank"}**
- **[RoBERTa for Question Answering](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-roberta-qa){:target="_blank"}**
- **[T5 for Abstractive Question Answering](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-t5-qa){:target="_blank"}**

## Useful Resources

Want to dive deeper into question answering with Spark NLP? Here are some curated resources to help you get started and explore further:

**Articles and Guides**
- *[Empowering NLP with Spark NLP and T5 Model: Text Summarization and Question Answering](https://www.johnsnowlabs.com/empowering-nlp-with-spark-nlp-and-t5-model-text-summarization-and-question-answering/){:target="_blank"}*
- *[Question Answering in Visual NLP: A Picture is Worth a Thousand Answers](https://medium.com/spark-nlp/question-answering-in-visual-nlp-a-picture-is-worth-a-thousand-answers-535bbcb53d3c){:target="_blank"}*
- *[Spark NLP: Unlocking the Power of Question Answering](https://medium.com/john-snow-labs/spark-nlp-unlocking-the-power-of-question-answering-e5a60f925368){:target="_blank"}*

**Notebooks**
- *[Question Answering Transformers in Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/22.0_Llama2_Transformer_In_SparkNLP.ipynb){:target="_blank"}*
- *[Question Answering and Summarization with T5](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/10.Question_Answering_and_Summarization_with_T5.ipynb){:target="_blank"}*
- *[Question Answering in Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/annotation/text/english/question-answering){:target="_blank"}*
- *[T5 Workshop with Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/10.1_T5_Workshop_with_Spark_NLP.ipynb){:target="_blank"}*
