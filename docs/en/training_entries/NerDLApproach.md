{%- capture title -%}
NerDLApproach
{%- endcapture -%}

{%- capture description -%}
This Named Entity recognition annotator allows to train generic NER model based on Neural Networks.

The architecture of the neural network is a Char CNNs - BiLSTM - CRF that achieves state-of-the-art in most datasets.

The training data should be a labeled Spark Dataset, in the format of [CoNLL](/docs/en/training#conll-dataset)
2003 IOB with `Annotation` type columns. The data should have columns of type `DOCUMENT, TOKEN, WORD_EMBEDDINGS` and an
additional label column of annotator type `NAMED_ENTITY`.
Excluding the label, this can be done with for example
  - a [SentenceDetector](/docs/en/annotators#sentencedetector),
  - a [Tokenizer](/docs/en/annotators#tokenizer) and
  - a [WordEmbeddingsModel](/docs/en/annotators#wordembeddings)
  (any word embeddings can be chosen, e.g. [BertEmbeddings](/docs/en/transformers#bertembeddings) for BERT based embeddings).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter/training/english/dl-ner).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

# This CoNLL dataset already includes a sentence, token and label
# column with their respective annotator types. If a custom dataset is used,
# these need to be defined with for example:

documentAssembler = DocumentAssembler() \\
    .setInputCol("text") \\
    .setOutputCol("document")

sentence = SentenceDetector() \\
    .setInputCols(["document"]) \\
    .setOutputCol("sentence")

tokenizer = Tokenizer() \\
    .setInputCols(["sentence"]) \\
    .setOutputCol("token")

Then the training can start

embeddings = BertEmbeddings.pretrained() \\
    .setInputCols(["sentence", "token"]) \\
    .setOutputCol("embeddings")

nerTagger = NerDLApproach() \\
    .setInputCols(["sentence", "token", "embeddings"]) \\
    .setLabelColumn("label") \\
    .setOutputCol("ner") \\
    .setMaxEpochs(1) \\
    .setRandomSeed(0) \\
    .setVerbose(0)

pipeline = Pipeline().setStages([
    embeddings,
    nerTagger
])

# We use the sentences, tokens, and labels from the CoNLL dataset.

conll = CoNLL()
trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")
pipelineModel = pipeline.fit(trainingData)
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.BertEmbeddings
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.ml.Pipeline

// This CoNLL dataset already includes a sentence, token and label
// column with their respective annotator types. If a custom dataset is used,
// these need to be defined with for example:

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

// Then the training can start
val embeddings = BertEmbeddings.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("embeddings")

val nerTagger = new NerDLApproach()
  .setInputCols("sentence", "token", "embeddings")
  .setLabelColumn("label")
  .setOutputCol("ner")
  .setMaxEpochs(1)
  .setRandomSeed(0)
  .setVerbose(0)

val pipeline = new Pipeline().setStages(Array(
  embeddings,
  nerTagger
))

// We use the sentences, tokens and labels from the CoNLL dataset
val conll = CoNLL()
val trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")

val pipelineModel = pipeline.fit(trainingData)
{%- endcapture -%}

{%- capture api_link -%}
[NerDLApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[NerDLApproach](/api/python/reference/autosummary/python/sparknlp/annotator/ner/ner_dl/index.html#sparknlp.annotator.ner.ner_dl.NerDLApproach)
{%- endcapture -%}

{%- capture source_link -%}
[NerDLApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLApproach.scala)
{%- endcapture -%}

{% include templates/training_anno_template.md
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
