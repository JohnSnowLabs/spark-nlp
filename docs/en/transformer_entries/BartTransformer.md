{%- capture title -%}
BartTransformer
{%- endcapture -%}

{%- capture description -%}
BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation,
Translation, and Comprehension Transformer

The Facebook BART (Bidirectional and Auto-Regressive Transformer) model is a state-of-the-art
language generation model that was introduced by Facebook AI in 2019. It is based on the
transformer architecture and is designed to handle a wide range of natural language processing
tasks such as text generation, summarization, and machine translation.

BART is unique in that it is both bidirectional and auto-regressive, meaning that it can
generate text both from left-to-right and from right-to-left. This allows it to capture
contextual information from both past and future tokens in a sentence,resulting in more
accurate and natural language generation.

The model was trained on a large corpus of text data using a combination of unsupervised and
supervised learning techniques. It incorporates pretraining and fine-tuning phases, where the
model is first trained on a large unlabeled corpus of text, and then fine-tuned on specific
downstream tasks.

BART has achieved state-of-the-art performance on a wide range of NLP tasks, including
summarization, question-answering, and language translation. Its ability to handle multiple
tasks and its high performance on each of these tasks make it a versatile and valuable tool
for natural language processing applications.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val bart = BartTransformer.pretrained()
  .setInputCols("document")
  .setOutputCol("generation")
```

The default model is `"bart_large_cnn"`, if no name is provided. For available pretrained
models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?q=bart).

For extended examples of usage, see
[BartTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/BartTestSpec.scala).

**References:**

- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://aclanthology.org/2020.acl-main.703.pdf)
- https://github.com/pytorch/fairseq

**Paper Abstract:**

*We present BART, a denoising autoencoder for pretraining sequence-to-sequence models. BART
is trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model
to reconstruct the original text. It uses a standard Tranformer-based neural machine
translation architecture which, despite its simplicity, can be seen as generalizing BERT (due
to the bidirectional encoder), GPT (with the left-to-right decoder), and other recent
pretraining schemes. We evaluate a number of noising approaches, finding the best performance
by both randomly shuffling the order of sentences and using a novel in-filling scheme, where
spans of text are replaced with a single mask token. BART is particularly effective when fine
tuned for text generation but also works well for comprehension tasks. It matches the
performance of RoBERTa on GLUE and SQuAD, and achieves new stateof-the-art results on a range
of abstractive dialogue, question answering, and summarization tasks, with gains of up to 3.5
ROUGE. BART also provides a 1.1 BLEU increase over a back-translation system for machine
translation, with only target language pretraining. We also replicate other pretraining
schemes within the BART framework, to understand their effect on end-task performance*

**Note:**

This is a very computationally expensive module especially on larger sequence. The use of an
accelerator such as GPU is recommended.
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
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
bart = BartTransformer.pretrained("bart_large_cnn") \
    .setTask("summarize:") \
    .setInputCols(["documents"]) \
    .setMaxOutputLength(200) \
    .setOutputCol("summaries")

pipeline = Pipeline().setStages([documentAssembler, bart])

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
import com.johnsnowlabs.nlp.annotators.seq2seq.GPT2Transformer
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

val bart = BartTransformer.pretrained("bart_large_cnn")
  .setInputCols(Array("documents"))
  .setMinOutputLength(10)
  .setMaxOutputLength(30)
  .setDoSample(true)
  .setTopK(50)
  .setOutputCol("generation")

val pipeline = new Pipeline().setStages(Array(documentAssembler, bart))

val data = Seq(
  "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
  "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
  "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
).toDF("text")
val result = pipeline.fit(data).transform(data)

results.select("generation.result").show(truncate = false)
+--------------------------------------------------------------+
|result                                                        |
+--------------------------------------------------------------+
|[Nearly 800 thousand customers were affected by the shutoffs.]|
+--------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[BartTransformer](/api/com/johnsnowlabs/nlp/annotators/seq2seq/BartTransformer)
{%- endcapture -%}

{%- capture python_api_link -%}
[BartTransformer](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/bart_transformer/index.html#sparknlp.annotator.seq2seq.bart_transformer.BartTransformer)
{%- endcapture -%}

{%- capture source_link -%}
[BartTransformer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/BartTransformer.scala)
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