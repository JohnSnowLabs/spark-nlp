{%- capture title -%}
NerDLGraphChecker
{%- endcapture -%}

{%- capture description -%}
Checks whether a suitable NerDL graph is available for the given training dataset, before any
computations/training is done. This annotator is useful for custom training cases, where
specialized graphs might not be available and we want to check before embeddings are evaluated.

Important: This annotator should be used or positioned before any embedding or NerDLApproach
annotators in the pipeline and will process the whole dataset to extract the required graph parameters.

This annotator requires a dataset with at least two columns: one with tokens and one with the
labels. In addition, it requires the used embedding annotator in the pipeline to extract the
suitable embedding dimension.

For extended examples of usage, see the
[example notebook](https://github.com/JohnSnowLabs/spark-nlp/blob/master//home/ducha/Workspace/scala/spark-nlp-feature/examples/python/training/english/dl-ner/ner_dl_graph_checker.ipynb)
and the
[NerDLGraphCheckerTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLGraphCheckerTestSpec.scala).

{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
NONE
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import*
from pyspark.ml import Pipeline
conll = CoNLL()
trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")
embeddings = BertEmbeddings \
    .pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")
nerDLGraphChecker = NerDLGraphChecker() \
    .setInputCols(["sentence", "token"]) \
    .setLabelColumn("label") \
    .setEmbeddingsModel(embeddings)
nerTagger = NerDLApproach() \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setLabelColumn("label") \
    .setOutputCol("ner") \
    .setMaxEpochs(1) \
    .setRandomSeed(0) \
    .setVerbose(0)
pipeline = Pipeline().setStages([nerDLGraphChecker, embeddings, nerTagger])
# will throw an exception if no suitable graph found
pipelineModel = pipeline.fit(trainingData) 
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.ml.Pipeline

// This CoNLL dataset already includes a sentence, token and label
// column with their respective annotator types. If a custom dataset is used,
// these need to be defined with for example:
val conll = CoNLL()
val trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")

val embeddings = BertEmbeddings
  .pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("embeddings")

// Requires the data for NerDLApproach graphs: text, tokens, labels and the embedding model
val nerDLGraphChecker = new NerDLGraphChecker()
  .setInputCols("sentence", "token")
  .setLabelColumn("label")
  .setEmbeddingsModel(embeddings)

val nerTagger = new NerDLApproach()
  .setInputCols("sentence", "token", "embeddings")
  .setLabelColumn("label")
  .setOutputCol("ner")
  .setMaxEpochs(1)
  .setRandomSeed(0)
  .setVerbose(0)

val pipeline = new Pipeline().setStages(
  Array(nerDLGraphChecker, embeddings, nerTagger))

// Will throw an exception if no suitable graph is found
val pipelineModel = pipeline.fit(trainingData)
{%- endcapture -%}

{%- capture api_link -%}
[NerDLGraphChecker](/api/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLGraphChecker)
{%- endcapture -%}

{%- capture python_api_link -%}
[NerDLGraphChecker](/api/python/reference/autosummary/sparknlp/annotator/ner/ner_dl_graph_checker/index.html#sparknlp.annotator.ner.ner_dl_graph_checker.NerDLGraphChecker)
{%- endcapture -%}

{%- capture source_link -%}
[NerDLGraphChecker](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLGraphChecker.scala)
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
