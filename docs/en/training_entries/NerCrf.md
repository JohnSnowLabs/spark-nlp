{%- capture title -%}
NerCrfApproach
{%- endcapture -%}

{%- capture description -%}
Algorithm for training a Named Entity Recognition Model

This Named Entity recognition annotator allows for a generic model to be trained by utilizing a CRF machine learning
algorithm. The training data should be a labeled Spark Dataset, e.g. [CoNLL](/docs/en/training#conll-dataset) 2003 IOB with
`Annotation` type columns. The data should have columns of type `DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS` and an
additional label column of annotator type `NAMED_ENTITY`.
Excluding the label, this can be done with for example
  - a [SentenceDetector](/docs/en/annotators#sentencedetector),
  - a [Tokenizer](/docs/en/annotators#tokenizer) and
  - a [PerceptronModel](/docs/en/annotators#postagger-part-of-speech-tagger) and
  - a [WordEmbeddingsModel](/docs/en/annotators#wordembeddings)
  (any word embeddings can be chosen, e.g. [BertEmbeddings](/docs/en/transformers#bertembeddings) for BERT based embeddings).

Optionally the user can provide an entity dictionary file with `setExternalFeatures` for better accuracy.

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/crf-ner/ner_dl_crf.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture python_example -%}
# This CoNLL dataset already includes the sentence, token, pos and label column with their respective annotator types.
# If a custom dataset is used, these need to be defined.

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

embeddings = WordEmbeddingsModel.pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(False)

nerTagger = NerCrfApproach() \
    .setInputCols(["sentence", "token", "pos", "embeddings"]) \
    .setLabelColumn("label") \
    .setMinEpochs(1) \
    .setMaxEpochs(3) \
    .setC0(34) \
    .setL2(3.0) \
    .setOutputCol("ner")

pipeline = Pipeline().setStages([
    documentAssembler,
    embeddings,
    nerTagger
])


conll = CoNLL()
trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")

pipelineModel = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture scala_example -%}
// This CoNLL dataset already includes the sentence, token, pos and label column with their respective annotator types.
// If a custom dataset is used, these need to be defined.

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.annotator.NerCrfApproach
import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = WordEmbeddingsModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("embeddings")
  .setCaseSensitive(false)

val nerTagger = new NerCrfApproach()
  .setInputCols("sentence", "token", "pos", "embeddings")
  .setLabelColumn("label")
  .setMinEpochs(1)
  .setMaxEpochs(3)
  .setC0(34)
  .setL2(3.0)
  .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  embeddings,
  nerTagger
))


val conll = CoNLL()
val trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")

val pipelineModel = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture api_link -%}
[NerCrfApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[NerCrfApproach](https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.NerCrfApproach.html)
{%- endcapture -%}

{%- capture source_link -%}
[NerCrfApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfApproach.scala)
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
