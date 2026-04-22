{%- capture title -%}
ModernBertEmbeddings
{%- endcapture -%}

{%- capture description -%}
Token-level embeddings using ModernBERT (Modern Bidirectional Encoder Representations from Transformers) a state-of-the-art encoder model designed for improved efficiency and performance compared to traditional BERT
models. It incorporates modern improvements including Flash Attention, unpadding, and GeGLU activation functions,
and supports sequence lengths up to 8192 tokens.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val embeddings = ModernBertEmbeddings.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("modernbert_embeddings")
```
The default model is `"modernbert-base"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?task=Embeddings).

For extended examples of usage, see [ModernBertEmbeddings.ipynb](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/annotation/text/english/embeddings/ModernBertEmbeddings.ipynb) and [ModernBertEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/ModernBertEmbeddingsTestSpec.scala).

**Sources** :

[Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663)

[https://huggingface.co/answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)

**Paper abstract**

*Encoder-only transformer models such as BERT offer a great performance-size tradeoff for retrieval and
classification tasks with respect to larger decoder-only models. Despite being the workhorse of numerous
production pipelines, there have been limited Pareto improvements to BERT since its release. In this paper, we
introduce ModernBERT, bringing modern model optimizations to encoder-only models and representing a major Pareto
improvement over older encoders. Trained on 2 trillion tokens with a native 8192 sequence length, ModernBERT
models exhibit state-of-the-art results on a large pool of evaluations encompassing diverse classification tasks
and both single and multi-vector retrieval on different domains (including code). In addition to strong downstream
performance, ModernBERT is also the most speed and memory efficient encoder and is designed for inference on
common GPUs.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture api_link -%}
[ModernBertEmbeddings](/api/com/johnsnowlabs/nlp/embeddings/ModernBertEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[ModernBertEmbeddings](/api/python/reference/autosummary/sparknlp/annotator/embeddings/modernbert_embeddings/index.html#sparknlp.annotator.embeddings.modernbert_embeddings.ModernBertEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[ModernBertEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/ModernBertEmbeddings.scala)
{%- endcapture -%}

{%- capture prediction_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# First extract the prerequisites for the NerDLModel
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

# Use the transformer embeddings
embeddings = ModernBertEmbeddings.pretrained() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")

# This pretrained model requires those specific transformer embeddings
ner_model = NerDLModel.pretrained("ner_dl_bert", "en") \
    .setInputCols(["document", "token", "embeddings"]) \
    .setOutputCol("ner")

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    tokenizer,
    embeddings,
    ner_model
])

data = spark.createDataFrame([["U.N. official Ekeus heads for Baghdad."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("ner.result").show(truncate=False)
+------------------------------------+
|result                              |
+------------------------------------+
|[I-LOC, O, O, I-PER, O, O, I-LOC, O]|
+------------------------------------+
{%- endcapture -%}

{%- capture prediction_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.ModernBertEmbeddings
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import org.apache.spark.ml.Pipeline

// First extract the prerequisites for the NerDLModel
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

// Use the transformer embeddings
val embeddings = ModernBertEmbeddings.pretrained()
  .setInputCols(Array("document", "token"))
  .setOutputCol("embeddings")

// This pretrained model requires those specific transformer embeddings
val nerModel = NerDLModel.pretrained("ner_dl_bert", "en")
  .setInputCols(Array("document", "token", "embeddings"))
  .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  embeddings,
  nerModel
))

val data = Seq("U.N. official Ekeus heads for Baghdad.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("ner.result").show(false)
+------------------------------------+
|result                              |
+------------------------------------+
|[I-LOC, O, O, I-PER, O, O, I-LOC, O]|
+------------------------------------+
{%- endcapture -%}

{%- capture training_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

# First extract the prerequisites for the NerDLApproach
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

embeddings = ModernBertEmbeddings.pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

# Then the training can start with the transformer embeddings
nerTagger = NerDLApproach() \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setLabelColumn("label") \
    .setOutputCol("ner") \
    .setMaxEpochs(1) \
    .setVerbose(0)

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    tokenizer,
    embeddings,
    nerTagger
])

# We use the text and labels from the CoNLL dataset
conll = CoNLL()
trainingData = conll.readDataset(spark, "eng.train")

pipelineModel = pipeline.fit(trainingData)
{%- endcapture -%}

{%- capture training_scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.ModernBertEmbeddings
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.ml.Pipeline

// First extract the prerequisites for the NerDLApproach
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = ModernBertEmbeddings.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("embeddings")

// Then the training can start with the transformer embeddings
val nerTagger = new NerDLApproach()
  .setInputCols("sentence", "token", "embeddings")
  .setLabelColumn("label")
  .setOutputCol("ner")
  .setMaxEpochs(1)
  .setVerbose(0)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  embeddings,
  nerTagger
))

// We use the text and labels from the CoNLL dataset
val conll = CoNLL()
val trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")

val pipelineModel = pipeline.fit(trainingData)
{%- endcapture -%}

{%- capture embeddings_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

embeddings = ModernBertEmbeddings.pretrained() \
    .setInputCols(["token", "document"]) \
    .setOutputCol("modernbert_embeddings")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["modernbert_embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True)

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    embeddings,
    embeddingsFinisher
])

data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[-2.3497989177703857,0.480538547039032,-0.3238905668258667,-1.612930893898010...|
|[-2.1357314586639404,0.32984697818756104,-0.6032363176345825,-1.6791689395904...|
|[-1.8244884014129639,-0.27088963985443115,-1.059438943862915,-0.9817547798156...|
|[-1.1648050546646118,-0.4725411534309387,-0.5938255786895752,-1.5780693292617...|
|[-0.9125322699546814,0.4563939869403839,-0.3975459933280945,-1.81611204147338...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture embeddings_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.ModernBertEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val embeddings = ModernBertEmbeddings.pretrained("modernbert-base", "en")
  .setInputCols("token", "document")
  .setOutputCol("modernbert_embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("modernbert_embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  embeddings,
  embeddingsFinisher
))

val data = Seq("This is a sentence.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[-2.3497989177703857,0.480538547039032,-0.3238905668258667,-1.612930893898010...|
|[-2.1357314586639404,0.32984697818756104,-0.6032363176345825,-1.6791689395904...|
|[-1.8244884014129639,-0.27088963985443115,-1.059438943862915,-0.9817547798156...|
|[-1.1648050546646118,-0.4725411534309387,-0.5938255786895752,-1.5780693292617...|
|[-0.9125322699546814,0.4563939869403839,-0.3975459933280945,-1.81611204147338...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{% include templates/transformer_usecases_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_api_link=python_api_link
api_link=api_link
source_link=source_link
prediction_python_example=prediction_python_example
prediction_scala_example=prediction_scala_example
training_python_example=training_python_example
training_scala_example=training_scala_example
embeddings_python_example=embeddings_python_example
embeddings_scala_example=embeddings_scala_example
%}
