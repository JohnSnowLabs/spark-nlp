{%- capture title -%}
ElmoEmbeddings
{%- endcapture -%}

{%- capture description -%}
Word embeddings from ELMo (Embeddings from Language Models), a language model trained on the 1 Billion Word Benchmark.

Note that this is a very computationally expensive module compared to word embedding modules that only perform
embedding lookups. The use of an accelerator is recommended.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val embeddings = ElmoEmbeddings.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("elmo_embeddings")

# Offline - Download the pretrained model manually and extract it
elmo = ElmoEmbeddings.load("/elmo_en_2.4.0_2.4_1580488815299") \
        .setInputCols("sentence", "token") \
        .setOutputCol("elmo")
```
The default model is `"elmo"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Embeddings).

The pooling layer can be set with `setPoolingLayer` to the following values:
  - `"word_emb"`: the character-based word representations with shape `[batch_size, max_length, 512]`.
  - `"lstm_outputs1"`: the first LSTM hidden state with shape `[batch_size, max_length, 1024]`.
  - `"lstm_outputs2"`: the second LSTM hidden state with shape `[batch_size, max_length, 1024]`.
  - `"elmo"`: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape `[batch_size, max_length, 1024]`.

For extended examples of usage, see the
[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/dl-ner/ner_elmo.ipynb)
and the [ElmoEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/ElmoEmbeddingsTestSpec.scala).

**Sources:**

https://tfhub.dev/google/elmo/3

[Deep contextualized word representations](https://arxiv.org/abs/1802.05365)

**Paper abstract:**

*We introduce a new type of deep contextualized word representation that models both (1) complex characteristics of
word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model
polysemy). Our word vectors are learned functions of the internal states of a deep bidirectional language model
(biLM), which is pre-trained on a large text corpus. We show that these representations can be easily added to
existing models and significantly improve the state of the art across six challenging NLP problems, including
question answering, textual entailment and sentiment analysis. We also present an analysis showing that exposing the
deep internals of the pre-trained network is crucial, allowing downstream models to mix different types of
semi-supervision signals.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture api_link -%}
[ElmoEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/ElmoEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[ElmoEmbeddings](/api/python/reference/autosummary/python/sparknlp/annotator/embeddings/elmo_embeddings/index.html#sparknlp.annotator.embeddings.elmo_embeddings.ElmoEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[ElmoEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/ElmoEmbeddings.scala)
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

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

# Use the transformer embeddings
embeddings = ElmoEmbeddings.pretrained() \
    .setInputCols(['document', 'token']) \
    .setOutputCol('embeddings')

# This pretrained model requires those specific transformer embeddings
ner_model = NerDLModel.pretrained("ner_conll_elmo", "en") \
    .setInputCols(["document", "token", "embeddings"]) \
    .setOutputCol("ner")

pipeline = Pipeline().setStages([
    documentAssembler,
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
import com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import org.apache.spark.ml.Pipeline

// First extract the prerequisites for the NerDLModel
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

// Use the transformer embeddings
val embeddings = ElmoEmbeddings.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("embeddings")

// This pretrained model requires those specific transformer embeddings
val nerModel = NerDLModel.pretrained("ner_conll_elmo", "en")
  .setInputCols("document", "token", "embeddings")
  .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
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

embeddings = ElmoEmbeddings.pretrained() \
    .setPoolingLayer("word_emb") \
    .setInputCols(["token", "sentence"]) \
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
import com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings
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

val embeddings = ElmoEmbeddings.pretrained()
  .setPoolingLayer("word_emb")
  .setInputCols("token", "sentence")
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

embeddings = ElmoEmbeddings.pretrained() \
    .setPoolingLayer("word_emb") \
    .setInputCols(["token", "document"]) \
    .setOutputCol("embeddings")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

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
|[6.662458181381226E-4,-0.2541114091873169,-0.6275503039360046,0.5787073969841...|
|[0.19154725968837738,0.22998669743537903,-0.2894386649131775,0.21524395048618...|
|[0.10400570929050446,0.12288510054349899,-0.07056470215320587,-0.246389418840...|
|[0.49932169914245605,-0.12706467509269714,0.30969417095184326,0.2643227577209...|
|[-0.8871506452560425,-0.20039963722229004,-1.0601330995559692,0.0348707810044...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture embeddings_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val embeddings = ElmoEmbeddings.pretrained()
  .setPoolingLayer("word_emb")
  .setInputCols("token", "document")
  .setOutputCol("embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)
  .setCleanAnnotations(false)

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
|[6.662458181381226E-4,-0.2541114091873169,-0.6275503039360046,0.5787073969841...|
|[0.19154725968837738,0.22998669743537903,-0.2894386649131775,0.21524395048618...|
|[0.10400570929050446,0.12288510054349899,-0.07056470215320587,-0.246389418840...|
|[0.49932169914245605,-0.12706467509269714,0.30969417095184326,0.2643227577209...|
|[-0.8871506452560425,-0.20039963722229004,-1.0601330995559692,0.0348707810044...|
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