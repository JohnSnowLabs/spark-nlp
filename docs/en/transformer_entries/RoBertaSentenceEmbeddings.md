{%- capture title -%}
RoBertaSentenceEmbeddings
{%- endcapture -%}

{%- capture description -%}
Sentence-level embeddings using RoBERTa. The RoBERTa model was proposed in [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
It is based on Google's BERT model released in 2018.

It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val embeddings = RoBertaSentenceEmbeddings.pretrained()
  .setInputCols("sentence")
  .setOutputCol("sentence_embeddings")
```
The default model is `"sent_roberta_base"`, if no name is provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Embeddings).

Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To see which models are compatible and how to import them see [Import Transformers into Spark NLP 🚀](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669).

**Paper Abstract:**

*Language model pretraining has led to significant performance gains but careful comparison between different
approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes,
and, as we will show, hyperparameter choices have significant impact on the final results. We present a replication
study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and
training data size. We find that BERT was significantly undertrained, and can match or exceed the performance of every
model published after it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results
highlight the importance of previously overlooked design choices, and raise questions about the source of recently
reported improvements. We release our models and code.*

Tips:
- RoBERTa has the same architecture as BERT, but uses a byte-level BPE as a tokenizer (same as GPT-2) and uses a different pretraining scheme.
- RoBERTa doesn't have :obj:`token_type_ids`, you don't need to indicate which token belongs to which segment. Just separate your segments with the separation token :obj:`tokenizer.sep_token` (or :obj:`</s>`)

The original code can be found ```here``` https://github.com/pytorch/fairseq/tree/master/examples/roberta.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture python_api_link -%}
[RoBertaSentenceEmbeddings](/api/python/reference/autosummary/python/sparknlp/annotator/embeddings/roberta_sentence_embeddings/index.html#sparknlp.annotator.embeddings.roberta_sentence_embeddings.RoBertaSentenceEmbeddings)
{%- endcapture -%}

{%- capture api_link -%}
[RoBertaSentenceEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/RoBertaSentenceEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[RoBertaSentenceEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/RoBertaSentenceEmbeddings.scala)
{%- endcapture -%}

{%- capture prediction_python_example -%}
# Coming Soon!
{%- endcapture -%}

{%- capture prediction_scala_example -%}
// Coming Soon!
{%- endcapture -%}

{%- capture training_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

smallCorpus = spark.read.option("header","True").csv("sentiment.csv")

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

embeddings = RoBertaSentenceEmbeddings.pretrained() \
  .setInputCols(["document"])\
  .setOutputCol("sentence_embeddings")

# Then the training can start with the transformer embeddings
docClassifier = ClassifierDLApproach() \
    .setInputCols("sentence_embeddings") \
    .setOutputCol("category") \
    .setLabelColumn("label") \
    .setBatchSize(64) \
    .setMaxEpochs(20) \
    .setLr(5e-3) \
    .setDropout(0.5)

pipeline = Pipeline().setStages([
    documentAssembler,
    embeddings,
    docClassifier
])

pipelineModel = pipeline.fit(smallCorpus)
{%- endcapture -%}

{%- capture training_scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.RoBertaSentenceEmbeddings
import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach
import org.apache.spark.ml.Pipeline

val smallCorpus = spark.read.option("header", "true").csv("src/test/resources/classifier/sentiment.csv")

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = RoBertaSentenceEmbeddings.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence_embeddings")

// Then the training can start with the transformer embeddings
val docClassifier = new ClassifierDLApproach()
  .setInputCols("sentence_embeddings")
  .setOutputCol("category")
  .setLabelColumn("label")
  .setBatchSize(64)
  .setMaxEpochs(20)
  .setLr(5e-3f)
  .setDropout(0.5f)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  embeddings,
  docClassifier
))

val pipelineModel = pipeline.fit(smallCorpus)
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

sentenceEmbeddings = RoBertaSentenceEmbeddings.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings") \
    .setCaseSensitive(True)

# you can either use the output to train ClassifierDL, SentimentDL, or MultiClassifierDL
# or you can use EmbeddingsFinisher to prepare the results for Spark ML functions

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      tokenizer,
      sentenceEmbeddings,
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
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("token")

val sentenceEmbeddings = RoBertaSentenceEmbeddings.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence_embeddings")
  .setCaseSensitive(true)

// you can either use the output to train ClassifierDL, SentimentDL, or MultiClassifierDL
// or you can use EmbeddingsFinisher to prepare the results for Spark ML functions

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("sentence_embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)
  .setCleanAnnotations(false)

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    tokenizer,
    sentenceEmbeddings,
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
