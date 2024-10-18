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

**Question Answering (QA)** is the task of automatically answering questions posed by humans in natural language. It is a fundamental problem in *natural language processing (NLP)*, playing a vital role in applications such as search engines, virtual assistants, customer support systems, and more. Spark NLP provides state-of-the-art (SOTA) models for QA tasks, enabling accurate and context-aware responses to user queries.

QA systems extract relevant information from a given context or knowledge base to answer a question. Depending on the model and input, they can either find exact answers within a text or generate a more comprehensive response.

## Types of Question Answering

- **Open-Book QA:** In this approach, the model has access to external documents, passages, or knowledge sources to extract the answer. The system looks for relevant information within the provided text (e.g., "What is the tallest mountain in the world?" answered using a document about mountains).
  
- **Closed-Book QA:** Here, the model must rely solely on the knowledge it has been trained on, without access to external sources. The answer is generated from the model's internal knowledge (e.g., answering trivia questions without referring to external material).

Common use cases include:

- **Fact-based QA:** Answering factoid questions such as "What is the capital of France?"
- **Reading Comprehension:** Extracting answers from a provided context, often used in assessments or educational tools.
- **Dialogue-based QA:** Supporting interactive systems that maintain context across multiple turns of conversation.

By leveraging QA models, organizations can build robust systems that improve user engagement, provide instant information retrieval, and offer customer support in a more intuitive manner.

<!-- <div style="text-align: center;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/dQw4w9WgXcQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div> -->

## Picking a Model

When selecting a model for question answering, consider the following important factors. First, assess the **nature of your data** (e.g., structured knowledge base vs. unstructured text) and the **type of QA** needed (open-book or closed-book). Open-book QA requires models that can efficiently search and extract from external sources, while closed-book QA demands models with a large internal knowledge base.

Evaluate the **complexity of the questions**—are they simple factoids or require more reasoning and multi-turn interactions? Metrics such as **Exact Match (EM)** and **F1 score** are commonly used to measure model performance in QA tasks. Finally, take into account the **computational resources** available, as some models, like BERT or T5, may require significant processing power.

Explore models tailored for question answering at [Spark NLP Models](https://sparknlp.org/models), where you’ll find various options for different QA tasks.

#### Recommended Models for Specific QA Tasks

- **Extractive QA:** Use models like [`distilbert-base-cased-distilled-squad`](https://sparknlp.org/2023/11/26/distilbert_base_cased_qa_squad2_en.html){:target="_blank"} and [`bert-large-uncased-whole-word-masking-finetuned-squad`](https://sparknlp.org/2024/09/01/bert_large_uncased_whole_word_masking_finetuned_squad_google_bert_en.html){:target="_blank"} for extracting answers directly from a provided context.
- **Generative QA (Closed-Book):** Consider models such as [`roberta-base-squad2`](https://sparknlp.org/2022/12/02/roberta_qa_deepset_base_squad2_en.html){:target="_blank"} or [`t5_base`](https://sparknlp.org/2021/01/08/t5_base_en.html){:target="_blank"} for generating answers based on internal knowledge without external context.

By selecting the appropriate question answering model, you can enhance your ability to deliver accurate and relevant answers tailored to your specific NLP tasks.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# 1. Document Assembler: Prepares the question and context text for further processing
documentAssembler = MultiDocumentAssembler() \
    .setInputCols(["question", "context"]) \
    .setOutputCol(["document_question", "document_context"])

# 2. Question Answering Model: Uses a pretrained RoBERTa model for QA
spanClassifier = RoBertaForQuestionAnswering.pretrained() \
    .setInputCols(["document_question", "document_context"]) \
    .setOutputCol("answer") \
    .setCaseSensitive(False)

# 3. Pipeline: Combines the stages (DocumentAssembler and RoBERTa model) into a pipeline
pipeline = Pipeline().setStages([
    documentAssembler,
    spanClassifier
])

# 4. Sample Data: Creating a DataFrame with a question and context
data = spark.createDataFrame([["What's my name?", "My name is Clara and I live in Berkeley."]]).toDF("question", "context")

# 5. Running the Pipeline: Fitting the pipeline to the data and generating answers
result = pipeline.fit(data).transform(data)

# 6. Displaying the Result: The output is the answer to the question extracted from the context
result.select("answer.result").show(truncate=False)

+--------------------+
|result              |
+--------------------+
|[Clara]             |
+--------------------+
```

```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

// 1. Document Assembler: Prepares the question and context text for further processing
val document = new MultiDocumentAssembler()
  .setInputCols("question", "context")
  .setOutputCols("document_question", "document_context")

// 2. Question Answering Model: Uses a pretrained RoBERTa model for QA
val questionAnswering = RoBertaForQuestionAnswering.pretrained()
  .setInputCols(Array("document_question", "document_context"))
  .setOutputCol("answer")
  .setCaseSensitive(true)

// 3. Pipeline: Combines the stages (DocumentAssembler and RoBERTa model) into a pipeline
val pipeline = new Pipeline().setStages(Array(
  document,
  questionAnswering
))

// 4. Sample Data: Creating a DataFrame with a question and context
val data = Seq("What's my name?", "My name is Clara and I live in Berkeley.").toDF("question", "context")

// 5. Running the Pipeline: Fitting the pipeline to the data and generating answers
val result = pipeline.fit(data).transform(data)

// 6. Displaying the Result: The output is the answer to the question extracted from the context
result.select("answer.result").show(false)

+---------------------+
|result               |
+---------------------+
|[Clara]              |
+---------------------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of question answering models in real time, visit our interactive demos:

- **[BERT for Extractive Question Answering](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-bert-qa){:target="_blank"}** – Extract answers directly from provided context using the BERT model.
- **[RoBERTa for Question Answering](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-roberta-qa){:target="_blank"}** – Use RoBERTa for advanced extractive question answering tasks.
- **[T5 for Abstractive Question Answering](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-t5-qa){:target="_blank"}** – Generate abstractive answers using Google's T5 model.
- **[Multihop QA with BERT](https://sparknlp.org/question_answering){:target="_blank"}** – Perform complex multihop question answering by reasoning over multiple pieces of text.

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
