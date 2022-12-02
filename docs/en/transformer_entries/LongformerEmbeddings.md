{%- capture title -%}
LongformerEmbeddings
{%- endcapture -%}

{%- capture description -%}
Longformer is a transformer model for long documents. The Longformer model was presented in [Longformer: The Long-Document Transformer](https://arxiv.org/pdf/2004.05150.pdf) by Iz Beltagy, Matthew E. Peters, Arman Cohan.
longformer-base-4096 is a BERT-like model started from the RoBERTa checkpoint and pretrained for MLM on long documents.
It supports sequences of length up to 4,096.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val embeddings = LongformerEmbeddings.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("embeddings")
```
The default model is `"longformer_base_4096"`, if no name is provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Embeddings).

For some examples of usage, see [LongformerEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/LongformerEmbeddingsTestSpec.scala).
Models from the HuggingFace 🤗 Transformers library are compatible with Spark NLP 🚀. To see which models are compatible and how to import them see [Import Transformers into Spark NLP 🚀](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669).

**Paper Abstract:**

*Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length.
To address this limitation, we introduce the Longformer with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer.
Longformer's attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention.
Following prior work on long-sequence transformers, we evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8.
In contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks.
Our pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on WikiHop and TriviaQA.
We finally introduce the Longformer-Encoder-Decoder (LED), a Longformer variant for supporting long document generative sequence-to-sequence tasks, and demonstrate its effectiveness on the arXiv summarization dataset.*

The original code can be found ```here``` https://github.com/allenai/longformer.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture api_link -%}
[LongformerEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/LongformerEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[LongformerEmbeddings](/api/python/reference/autosummary/python/sparknlp/annotator/embeddings/longformer_embeddings/index.html#sparknlp.annotator.embeddings.longformer_embeddings.LongformerEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[LongformerEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/LongformerEmbeddings.scala)
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
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

# Use the transformer embeddings
embeddings = LongformerEmbeddings \
      .pretrained("longformer_large_4096") \
      .setInputCols(['document', 'token']) \
      .setOutputCol("embeddings") \
      .setCaseSensitive(True) \
      .setMaxSentenceLength(4096)

# This pretrained model requires those specific transformer embeddings
ner_model = NerDLModel.pretrained('ner_conll_longformer_large_4096', 'en') \
    .setInputCols(['document', 'token', 'embeddings']) \
    .setOutputCol('ner')

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
|[B-ORG, O, O, B-PER, O, O, B-LOC, O]|
+------------------------------------+
{%- endcapture -%}

{%- capture prediction_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.LongformerEmbeddings
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import org.apache.spark.ml.Pipeline

// First extract the prerequisites for the NerDLModel
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

// Use the transformer embeddings
val embeddings = LongformerEmbeddings.pretrained("longformer_large_4096", "en")
  .setInputCols("document", "token")
  .setOutputCol("embeddings")
  .setCaseSensitive(true)
  .setMaxSentenceLength(4096)

// This pretrained model requires those specific transformer embeddings
val nerModel = NerDLModel.pretrained("ner_conll_longformer_large_4096", "en")
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
|[B-ORG, O, O, B-PER, O, O, B-LOC, O]|
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

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

embeddings = LongformerEmbeddings \
      .pretrained("longformer_base_4096") \
      .setInputCols(['document', 'token']) \
      .setOutputCol("embeddings") \
      .setCaseSensitive(True) \
      .setMaxSentenceLength(4096)

# Then the training can start with the transformer embeddings
nerTagger = NerDLApproach() \
    .setInputCols(["document", "token", "embeddings"]) \
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
import com.johnsnowlabs.nlp.embeddings.LongformerEmbeddings
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.ml.Pipeline

// First extract the prerequisites for the NerDLApproach
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = LongformerEmbeddings.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("embeddings")
  .setCaseSensitive(true)

// Then the training can start with the transformer embeddings
val nerTagger = new NerDLApproach()
  .setInputCols("document", "token", "embeddings")
  .setLabelColumn("label")
  .setOutputCol("ner")
  .setMaxEpochs(1)
  .setRandomSeed(0)
  .setVerbose(0)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
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
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

embeddings = LongformerEmbeddings.pretrained() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

pipeline = Pipeline() \
    .setStages([
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
|[0.18792399764060974,-0.14591649174690247,0.20547787845134735,0.1468472778797...|
|[0.22845706343650818,0.18073144555091858,0.09725798666477203,-0.0417917296290...|
|[0.07037967443466187,-0.14801117777824402,-0.03603338822722435,-0.17893412709...|
|[-0.08734266459941864,0.2486150562763214,-0.009067727252840996,-0.24408400058...|
|[0.22409197688102722,-0.4312366545200348,0.1401449590921402,0.356410235166549...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture embeddings_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("token")

val embeddings = LongformerEmbeddings.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("embeddings")
  .setCaseSensitive(true)

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)
  .setCleanAnnotations(false)

val pipeline = new Pipeline()
  .setStages(Array(
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
|[0.18792399764060974,-0.14591649174690247,0.20547787845134735,0.1468472778797...|
|[0.22845706343650818,0.18073144555091858,0.09725798666477203,-0.0417917296290...|
|[0.07037967443466187,-0.14801117777824402,-0.03603338822722435,-0.17893412709...|
|[-0.08734266459941864,0.2486150562763214,-0.009067727252840996,-0.24408400058...|
|[0.22409197688102722,-0.4312366545200348,0.1401449590921402,0.356410235166549...|
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