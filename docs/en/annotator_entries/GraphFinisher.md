{%- capture title -%}
GraphFinisher
{%- endcapture -%}

{%- capture description -%}
Helper class to convert the knowledge graph from GraphExtraction into a generic format, such as RDF.
{%- endcapture -%}

{%- capture input_anno -%}
NONE
{%- endcapture -%}

{%- capture output_anno -%}
NONE
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
# This is a continuation of the example of
# GraphExtraction. To see how the graph is extracted, see the
# documentation of that class.

graphFinisher = GraphFinisher() \
    .setInputCol("graph") \
    .setOutputCol("graph_finished")
    .setOutputAs[False]

finishedResult = graphFinisher.transform(result)
finishedResult.select("text", "graph_finished").show(truncate=False)
+-----------------------------------------------------+-----------------------------------------------------------------------+
|text                                                 |graph_finished                                                         |
+-----------------------------------------------------+-----------------------------------------------------------------------+
|You and John prefer the morning flight through Denver|(morning,flat,flight), (flight,flat,Denver)|
+-----------------------------------------------------+-----------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
// This is a continuation of the example of
// [[com.johnsnowlabs.nlp.annotators.GraphExtraction GraphExtraction]]. To see how the graph is extracted, see the
// documentation of that class.
import com.johnsnowlabs.nlp.GraphFinisher

val graphFinisher = new GraphFinisher()
  .setInputCol("graph")
  .setOutputCol("graph_finished")
  .setOutputAsArray(false)

val finishedResult = graphFinisher.transform(result)
finishedResult.select("text", "graph_finished").show(false)
+-----------------------------------------------------+-----------------------------------------------------------------------+
|text                                                 |graph_finished                                                         |
+-----------------------------------------------------+-----------------------------------------------------------------------+
|You and John prefer the morning flight through Denver|[[(prefer,nsubj,morning), (morning,flat,flight), (flight,flat,Denver)]]|
+-----------------------------------------------------+-----------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[GraphFinisher](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/GraphFinisher)
{%- endcapture -%}

{%- capture python_api_link -%}
[GraphFinisher](/api/python/reference/autosummary/python/sparknlp/base/graph_finisher/index.html#sparknlp.base.graph_finisher.GraphFinisher)
{%- endcapture -%}

{%- capture source_link -%}
[GraphFinisher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/GraphFinisher.scala)
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