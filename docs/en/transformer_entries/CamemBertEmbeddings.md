{%- capture title -%}
CamemBertEmbeddings
{%- endcapture -%}

{%- capture description -%}
The CamemBERT model was proposed in CamemBERT: a Tasty French Language Model by Louis Martin,
Benjamin Muller, Pedro Javier Ortiz Suárez, Yoann Dupont, Laurent Romary, Éric Villemonte de
la Clergerie, Djamé Seddah, and Benoît Sagot. It is based on Facebook’s RoBERTa model released
in 2019. It is a model trained on 138GB of French text.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val embeddings = CamemBertEmbeddings.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("camembert_embeddings")
```
The default model is `"camembert_base"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://nlp.johnsnowlabs.com/models?task=Embeddings).

For extended examples of usage, see the
[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/blogposts/3.NER_with_BERT.ipynb)
and the
[CamemBertEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/CamemBertEmbeddingsTestSpec.scala).
To see which models are compatible and how to import them see
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669.

**Sources** :

[CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894)

https://huggingface.co/camembert

**Paper abstract**

*Pretrained language models are now ubiquitous in Natural Language Processing. Despite their
success, most available models have either been trained on English data or on the
concatenation of data in multiple languages. This makes practical use of such models --in all
languages except English-- very limited. In this paper, we investigate the feasibility of
training monolingual Transformer-based language models for other languages, taking French as
an example and evaluating our language models on part-of-speech tagging, dependency parsing,
named entity recognition and natural language inference tasks. We show that the use of web
crawled data is preferable to the use of Wikipedia data. More surprisingly, we show that a
relatively small web crawled dataset (4GB) leads to results that are as good as those obtained
using larger datasets (130+GB). Our best performing model CamemBERT reaches or improves the
state of the art in all four downstream tasks.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
WORD_EMBEDDINGS
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

# Use the transformer embeddings
embeddings = CamemBertEmbeddings.pretrained() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)

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

{%- endcapture -%}

{%- capture training_scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.XlmRoBertaEmbeddings
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

val embeddings = CamemBertEmbeddings.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("embeddings")

// Then the training can start with the transformer embeddings
val nerTagger = new NerDLApproach()
  .setInputCols("sentence", "token", "embeddings")
  .setLabelColumn("label")
  .setOutputCol("ner")
  .setMaxEpochs(1)
  .setRandomSeed(0)
  .setVerbose(0)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  embeddings,
  nerTagger
))

{%- endcapture -%}

{%- capture embeddings_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \\
    .setInputCol("text") \\
    .setOutputCol("document")
tokenizer = Tokenizer() \\
    .setInputCols(["document"]) \\
    .setOutputCol("token")
embeddings = CamemBertEmbeddings.pretrained() \\
    .setInputCols(["token", "document"]) \\
    .setOutputCol("camembert_embeddings")
embeddingsFinisher = EmbeddingsFinisher() \\
    .setInputCols(["camembert_embeddings"]) \\
    .setOutputCols("finished_embeddings") \\
    .setOutputAsVector(True)
pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    embeddings,
    embeddingsFinisher
])

data = spark.createDataFrame([["C'est une phrase."]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[0.08442357927560806,-0.12863239645957947,-0.03835778683423996,0.200479581952...|
|[0.048462312668561935,0.12637358903884888,-0.27429091930389404,-0.07516729831...|
|[0.02690504491329193,0.12104076147079468,0.012526623904705048,-0.031543646007...|
|[0.05877285450696945,-0.08773420006036758,-0.06381352990865707,0.122621834278...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture embeddings_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.CamemBertEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val embeddings = CamemBertEmbeddings.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("camembert_embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("camembert_embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  embeddings,
  embeddingsFinisher
))

val data = Seq("C'est une phrase.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[0.08442357927560806,-0.12863239645957947,-0.03835778683423996,0.200479581952...|
|[0.048462312668561935,0.12637358903884888,-0.27429091930389404,-0.07516729831...|
|[0.02690504491329193,0.12104076147079468,0.012526623904705048,-0.031543646007...|
|[0.05877285450696945,-0.08773420006036758,-0.06381352990865707,0.122621834278...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[CamemBertEmbeddings](/api/com/johnsnowlabs/nlp/embeddings/CamemBertEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[CamemBertEmbeddings](/api/python/reference/autosummary/sparknlp/annotator/embeddings/camembert_embeddings/index.html#sparknlp.annotator.embeddings.camembert_embeddings.CamemBertEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[CamemBertEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/CamemBertEmbeddings.scala)
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