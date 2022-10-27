{%- capture title -%}
NerCrf
{%- endcapture -%}

{%- capture model_description -%}
Extracts Named Entities based on a CRF Model.

This Named Entity recognition annotator allows for a generic model to be trained by utilizing a CRF machine learning
algorithm. The data should have columns of type `DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS`.
These can be extracted with for example
  - a [SentenceDetector](/docs/en/annotators#sentencedetector),
  - a [Tokenizer](/docs/en/annotators#tokenizer) and
  - a [PerceptronModel](/docs/en/annotators#postagger-part-of-speech-tagger)

This is the instantiated model of the NerCrfApproach.
For training your own model, please see the documentation of that class.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val nerTagger = NerCrfModel.pretrained()
  .setInputCols("sentence", "token", "word_embeddings", "pos")
  .setOutputCol("ner"
```
The default model is `"ner_crf"`, if no name is provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/model-downloader/Running_Pretrained_pipelines.ipynb).
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture model_output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture model_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# First extract the prerequisites for the NerCrfModel
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("word_embeddings")

posTagger = PerceptronModel.pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos")

# Then NER can be extracted
nerTagger = NerCrfModel.pretrained() \
    .setInputCols(["sentence", "token", "word_embeddings", "pos"]) \
    .setOutputCol("ner")

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    tokenizer,
    embeddings,
    posTagger,
    nerTagger
])

data = spark.createDataFrame([["U.N. official Ekeus heads for Baghdad."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("ner.result").show(truncate=False)
+------------------------------------+
|result                              |
+------------------------------------+
|[I-ORG, O, O, I-PER, O, O, I-LOC, O]|
+------------------------------------+

{%- endcapture -%}

{%- capture model_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel
import org.apache.spark.ml.Pipeline

// First extract the prerequisites for the NerCrfModel
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("word_embeddings")

val posTagger = PerceptronModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("pos")

// Then NER can be extracted
val nerTagger = NerCrfModel.pretrained()
  .setInputCols("sentence", "token", "word_embeddings", "pos")
  .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  embeddings,
  posTagger,
  nerTagger
))

val data = Seq("U.N. official Ekeus heads for Baghdad.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("ner.result").show(false)
+------------------------------------+
|result                              |
+------------------------------------+
|[I-ORG, O, O, I-PER, O, O, I-LOC, O]|
+------------------------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[NerCrfModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[NerCrfModel](/api/python/reference/autosummary/python/sparknlp/annotator/ner/ner_crf/index.html#sparknlp.annotator.ner.ner_crf.NerCrfModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[NerCrfModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Algorithm for training a Named Entity Recognition Model

For instantiated/pretrained models, see NerCrfModel.

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

Optionally the user can provide an entity dictionary file with setExternalFeatures for better accuracy.

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/crf-ner/ner_dl_crf.ipynb)
and the [NerCrfApproachTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfApproachTestSpec.scala).
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture approach_python_example -%}
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

{%- capture approach_scala_example -%}
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

{%- capture approach_api_link -%}
[NerCrfApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[NerCrfApproach](/api/python/reference/autosummary/python/sparknlp/annotator/ner/ner_crf/index.html#sparknlp.annotator.ner.ner_crf.NerCrfApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[NerCrfApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfApproach.scala)
{%- endcapture -%}


{% include templates/approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_api_link=model_python_api_link
model_api_link=model_api_link
model_source_link=model_source_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_python_api_link=approach_python_api_link
approach_api_link=approach_api_link
approach_source_link=approach_source_link
%}
