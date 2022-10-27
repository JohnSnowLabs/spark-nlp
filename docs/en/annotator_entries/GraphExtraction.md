{%- capture title -%}
GraphExtraction
{%- endcapture -%}

{%- capture description -%}
Extracts a dependency graph between entities.

The GraphExtraction class takes e.g. extracted entities from a
[NerDLModel](/docs/en/annotators#nerdl) and creates a dependency tree which describes how
the entities relate to each other. For that a triple store format is used. Nodes represent the entities and the
edges represent the relations between those entities. The graph can then be used to find relevant relationships
between words.

Both the [DependencyParserModel](/docs/en/annotators#dependencyparser) and
[TypedDependencyParserModel](/docs/en/annotators#typeddependencyparser) need to be
present in the pipeline. There are two ways to set them:

  1. Both Annotators are present in the pipeline already. The dependencies are taken implicitly from these two
     Annotators.
  1. Setting `setMergeEntities` to `true` will download the default pretrained models for those two Annotators
     automatically. The specific models can also be set with `setDependencyParserModel` and
     `setTypedDependencyParserModel`:
     ```
           val graph_extraction = new GraphExtraction()
             .setInputCols("document", "token", "ner")
             .setOutputCol("graph")
             .setRelationshipTypes(Array("prefer-LOC"))
             .setMergeEntities(true)
           //.setDependencyParserModel(Array("dependency_conllu", "en",  "public/models"))
           //.setTypedDependencyParserModel(Array("dependency_typed_conllu", "en",  "public/models"))
     ```

To transform the resulting graph into a more generic form such as RDF, see the
[GraphFinisher](/docs/en/annotators#graphfinisher).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN, NAMED_ENTITY
{%- endcapture -%}

{%- capture output_anno -%}
NODE
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

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
    .setOutputCol("embeddings")

nerTagger = NerDLModel.pretrained() \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

posTagger = PerceptronModel.pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos")

dependencyParser = DependencyParserModel.pretrained() \
    .setInputCols(["sentence", "pos", "token"]) \
    .setOutputCol("dependency")

typedDependencyParser = TypedDependencyParserModel.pretrained() \
    .setInputCols(["dependency", "pos", "token"]) \
    .setOutputCol("dependency_type")

graph_extraction = GraphExtraction() \
    .setInputCols(["document", "token", "ner"]) \
    .setOutputCol("graph") \
    .setRelationshipTypes(["prefer-LOC"])

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    tokenizer,
    embeddings,
    nerTagger,
    posTagger,
    dependencyParser,
    typedDependencyParser,
    graph_extraction
])

data = spark.createDataFrame([["You and John prefer the morning flight through Denver"]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("graph").show(truncate=False)
+-----------------------------------------------------------------------------------------------------------------+
|graph                                                                                                            |
+-----------------------------------------------------------------------------------------------------------------+
|13, 18, prefer, [relationship -> prefer,LOC, path1 -> prefer,nsubj,morning,flat,flight,flat,Denver], []|
+-----------------------------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
import org.apache.spark.ml.Pipeline
import com.johnsnowlabs.nlp.annotators.GraphExtraction

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
  .setOutputCol("embeddings")

val nerTagger = NerDLModel.pretrained()
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")

val posTagger = PerceptronModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("pos")

val dependencyParser = DependencyParserModel.pretrained()
  .setInputCols("sentence", "pos", "token")
  .setOutputCol("dependency")

val typedDependencyParser = TypedDependencyParserModel.pretrained()
  .setInputCols("dependency", "pos", "token")
  .setOutputCol("dependency_type")

val graph_extraction = new GraphExtraction()
  .setInputCols("document", "token", "ner")
  .setOutputCol("graph")
  .setRelationshipTypes(Array("prefer-LOC"))

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  embeddings,
  nerTagger,
  posTagger,
  dependencyParser,
  typedDependencyParser,
  graph_extraction
))

val data = Seq("You and John prefer the morning flight through Denver").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("graph").show(false)
+-----------------------------------------------------------------------------------------------------------------+
|graph                                                                                                            |
+-----------------------------------------------------------------------------------------------------------------+
|[[node, 13, 18, prefer, [relationship -> prefer,LOC, path1 -> prefer,nsubj,morning,flat,flight,flat,Denver], []]]|
+-----------------------------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[GraphExtraction](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/GraphExtraction)
{%- endcapture -%}

{%- capture python_api_link -%}
[GraphExtraction](/api/python/reference/autosummary/python/sparknlp/annotator/graph_extraction/index.html#sparknlp.annotator.graph_extraction.GraphExtraction)
{%- endcapture -%}

{%- capture source_link -%}
[GraphExtraction](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/GraphExtraction.scala)
{%- endcapture -%}

{% include templates/anno_template.md
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