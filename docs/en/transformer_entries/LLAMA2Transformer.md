{%- capture title -%}
LLAMA2Transformer
{%- endcapture -%}

{%- capture description -%}
[Llama 2](https://huggingface.co/papers/2307.09288) is a family of large language models, Llama 2 and Llama 2-Chat, available in 7B, 13B, and 70B parameters. The Llama 2 model mostly keeps the same architecture as Llama, but it is pretrained on more tokens, doubles the context length, and uses grouped-query attention (GQA) in the 70B model to improve inference.

Llama 2-Chat is trained with supervised fine-tuning (SFT), and reinforcement learning with human feedback (RLHF) - rejection sampling and proximal policy optimization (PPO) - is applied to the fine-tuned model to align the chat model with human preferences.

Pretrained models can be loaded with `pretrained` of the companion object:
```scala
val llama2 = LLAMA2Transformer.pretrained("llama_2_7b_chat_hf_int4") 
    .setMaxOutputLength(50) 
    .setDoSample(False) 
    .setInputCols(["document"]) 
    .setOutputCol(["generation"])
```
The default model is `"llama_2_7b_chat_hf_int4"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=LLAMA2Transformer).

Spark NLP also supports Hugging Face transformer-based code generation models. Learn more here:  
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Resource**:

- [LLaMA 2 Paper on HuggingFace](https://huggingface.co/papers/2307.09288)
- [LLaMA 2 – Every Resource You Need (Philipp Schmid)](https://www.philschmid.de/llama-2)
- [Meta AI "5 Steps to Getting Started with Llama 2"](https://ai.meta.com/blog/5-steps-to-getting-started-with-llama-2/)
- [Awesome‑llama‑resources on GitHub](https://github.com/MIBlue119/awesome-llama-resources)
- [Fine‑Tuning Llama 2: Step‑by‑Step Guide (DataCamp)](https://www.datacamp.com/tutorial/fine-tuning-llama-2)
- [LLM Benchmarking with LLaMA2 (2025 Paper)](https://arxiv.org/abs/2503.19217)


**Paper abstract**

*In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
GENERATION
{%- endcapture -%}

{%- capture api_link -%}
[LLAMA2Transformer](/api/com/johnsnowlabs/nlp/annotators/seq2seq/LLAMA2Transformer.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[LLAMA2Transformer](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/llama2_transformer/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[LLAMA2Transformer](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/LLAMA2Transformer.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import LLAMA2Transformer
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

llama2 = LLAMA2Transformer.pretrained() \
    .setInputCols(["documents"]) \
    .setOutputCol("generation") \
    .setMinOutputLength(50) \
    .setMaxOutputLength(250) \
    .setDoSample(True) \
    .setTemperature(0.7) \
    .setTopK(50) \
    .setTopP(0.9) \
    .setRepetitionPenalty(1.1) \
    .setNoRepeatNgramSize(3) \
    .setIgnoreTokenIds([])

pipeline = Pipeline().setStages([
    document_assembler,
    llama2
])

prompt = spark.createDataFrame([("""
### System:
You are a concise assistant who explains machine learning in simple terms.

### User:
Explain the difference between supervised and unsupervised learning with examples.

### Assistant:
""",)], ["text"])

model = pipeline.fit(prompt)
results = model.transform(prompt)

results.select("generation.result").show(truncate=False)

+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[_\n### System:\nYou are a concise assistant who explains machine learning in simple terms.\n\n### User:\nExplain the difference between supervised and unsupervised learning with examples.\n\n### Assistant:\nOf course! Supervised learning is when you train a machine learning model on labeled data, where the correct output is already known. The model learns to predict the output based on the input data. For example, if you want to build a model that can recognize dogs and cats based on their pictures, you would need to provide the model with a dataset of labeled images of dogs and cat, where each image is associated with the correct label (dog or cat).\n\nOn the other hand, unsuperived learning is When you train machine learning models on unlabeled data. The goal is to identify patterns or relationships in the data without any prior knowledge of the expected output. For instance, if we want to analyze customer buying behavior, we can use unsupervise learning to identify common characteristics of customers who buy more products.\nIn summary, supervised learning relies on labeled data to train the model, while unsupervisied learning relied on unlabelled data.\nIs there anything else you would like to know?]|
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

val llama2 = LLaMA2Transformer.pretrained()
  .setInputCols("documents")
  .setOutputCol("generation")
  .setMinOutputLength(50)
  .setMaxOutputLength(250)
  .setDoSample(true)
  .setTemperature(0.7f)
  .setTopK(50)
  .setTopP(0.9f)
  .setRepetitionPenalty(1.1f)
  .setNoRepeatNgramSize(3)
  .setIgnoreTokenIds(Array.emptyIntArray)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  llama2
))

val prompt = Seq("""
### System:
You are a concise assistant who explains machine learning in simple terms.

### User:
Explain the difference between supervised and unsupervised learning with examples.

### Assistant:
""").toDF("text")

val model = pipeline.fit(prompt)
val results = model.transform(prompt)

results.select("generation.result").show(false)

+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[_\n### System:\nYou are a concise assistant who explains machine learning in simple terms.\n\n### User:\nExplain the difference between supervised and unsupervised learning with examples.\n\n### Assistant:\nOf course! Supervised learning is when you train a machine learning model on labeled data, where the correct output is already known. The model learns to predict the output based on the input data. For example, if you want to build a model that can recognize dogs and cats based on their pictures, you would need to provide the model with a dataset of labeled images of dogs and cat, where each image is associated with the correct label (dog or cat).\n\nOn the other hand, unsuperived learning is When you train machine learning models on unlabeled data. The goal is to identify patterns or relationships in the data without any prior knowledge of the expected output. For instance, if we want to analyze customer buying behavior, we can use unsupervise learning to identify common characteristics of customers who buy more products.\nIn summary, supervised learning relies on labeled data to train the model, while unsupervisied learning relied on unlabelled data.\nIs there anything else you would like to know?]|
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}