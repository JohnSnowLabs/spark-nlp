{%- capture title -%}
T5Transformer
{%- endcapture -%}

{%- capture description -%}
T5: the Text-To-Text Transfer Transformer

T5 reconsiders all NLP tasks into a unified text-to-text-format where the input and output are always
text strings, in contrast to BERT-style models that can only output either a class label or a span of the input.
The text-to-text framework is able to use the same model, loss function, and hyper-parameters on any NLP task,
including machine translation, document summarization, question answering, and classification tasks
(e.g., sentiment analysis). T5 can even apply to regression tasks by training it to predict the string
representation of a number instead of the number itself.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val t5 = T5Transformer.pretrained()
  .setTask("summarize:")
  .setInputCols("document")
  .setOutputCol("summaries")
```
The default model is `"t5_small"`, if no name is provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?q=t5).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/10.Question_Answering_and_Summarization_with_T5.ipynb)
and the [T5TestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/T5TestSpec.scala).

**Sources:**
 - [Exploring Transfer Learning with T5: the Text-To-Text Transfer Transformer](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)
 - [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
 - https://github.com/google-research/text-to-text-transfer-transformer

**Paper Abstract:**

*Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream
task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer
learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the
landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based
language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures,
unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining
the insights from our exploration with scale and our new Colossal Clean Crawled Corpus, we achieve state-of-the-art
results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate
future work on transfer learning for NLP, we release our data set, pre-trained models, and code.*

**Note:**

This is a very computationally expensive module especially on larger sequence.
The use of an accelerator such as GPU is recommended.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

t5 = T5Transformer.pretrained("t5_small") \
    .setTask("summarize:") \
    .setInputCols(["documents"]) \
    .setMaxOutputLength(200) \
    .setOutputCol("summaries")

pipeline = Pipeline().setStages([documentAssembler, t5])

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
result = pipeline.fit(data).transform(data)

result.select("summaries.result").show(truncate=False)
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                                                        |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[transfer learning has emerged as a powerful technique in natural language processing (NLP) the effectiveness of transfer learning has given rise to a diversity of approaches, methodologies, and practice .]|
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

val t5 = T5Transformer.pretrained("t5_small")
  .setTask("summarize:")
  .setInputCols(Array("documents"))
  .setMaxOutputLength(200)
  .setOutputCol("summaries")

val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

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
val result = pipeline.fit(data).transform(data)

result.select("summaries.result").show(false)
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                                                        |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[transfer learning has emerged as a powerful technique in natural language processing (NLP) the effectiveness of transfer learning has given rise to a diversity of approaches, methodologies, and practice .]|
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[T5Transformer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/seq2seq/T5Transformer)
{%- endcapture -%}

{%- capture python_api_link -%}
[T5Transformer](/api/python/reference/autosummary/python/sparknlp/annotator/seq2seq/t5_transformer/index.html#sparknlp.annotator.seq2seq.t5_transformer.T5Transformer)
{%- endcapture -%}

{%- capture source_link -%}
[T5Transformer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/T5Transformer.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
source_link=source_link
%}