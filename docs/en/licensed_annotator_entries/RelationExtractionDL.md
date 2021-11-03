{%- capture title -%}
RelationExtractionDL
{%- endcapture -%}

{%- capture description -%}
Extracts and classifies instances of relations between named entities.
In contrast with RelationExtractionModel, RelationExtractionDLModel is based on BERT.
For pretrained models please see the
[Models Hub](https://nlp.johnsnowlabs.com/models?task=Relation+Extraction) for available models.
{%- endcapture -%}

{%- capture input_anno -%}
CHUNK, DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
# Relation Extraction between body parts
# This is a continuation of the RENerChunksFilter example. See that class on how to extract the relation chunks.
# Define the extraction model
re_ner_chunk_filter = RENerChunksFilter() \
 .setInputCols(["ner_chunks", "dependencies"]) \
 .setOutputCol("re_ner_chunks") \
 .setMaxSyntacticDistance(4) \
 .setRelationPairs(["internal_organ_or_component-direction"])

re_model = RelationExtractionDLModel.pretrained("redl_bodypart_direction_biobert", "en", "clinical/models") \
  .setPredictionThreshold(0.5) \
  .setInputCols(["re_ner_chunks", "sentences"]) \
  .setOutputCol("relations")

trained_pipeline = Pipeline(stages=[
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  clinical_ner_tagger,
  ner_chunker,
  dependency_parser,
  re_ner_chunk_filter,
  re_model
])

data = spark.createDataFrame([["MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia"]]).toDF("text")
result = trained_pipeline.fit(data).transform(data)

# Show results
result.selectExpr("explode(relations) as relations") \
 .select(
   "relations.metadata.chunk1",
   "relations.metadata.entity1",
   "relations.metadata.chunk2",
   "relations.metadata.entity2",
   "relations.result"
 ) \
 .where("result != 0") \
 .show(truncate=False)
+------+---------+-------------+---------------------------+------+
|chunk1|entity1  |chunk2       |entity2                    |result|
+------+---------+-------------+---------------------------+------+
|upper |Direction|brain stem   |Internal_organ_or_component|1     |
|left  |Direction|cerebellum   |Internal_organ_or_component|1     |
|right |Direction|basil ganglia|Internal_organ_or_component|1     |
+------+---------+-------------+---------------------------+------+
{%- endcapture -%}

{%- capture scala_example -%}
// Relation Extraction between body parts
// This is a continuation of the [[RENerChunksFilter]] example. See that class on how to extract the relation chunks.
// Define the extraction model
val re_ner_chunk_filter = new RENerChunksFilter()
 .setInputCols("ner_chunks", "dependencies")
 .setOutputCol("re_ner_chunks")
 .setMaxSyntacticDistance(4)
 .setRelationPairs(Array("internal_organ_or_component-direction"))

val re_model = RelationExtractionDLModel.pretrained("redl_bodypart_direction_biobert", "en", "clinical/models")
  .setPredictionThreshold(0.5f)
  .setInputCols("re_ner_chunks", "sentences")
  .setOutputCol("relations")

val trained_pipeline = new Pipeline().setStages(Array(
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  clinical_ner_tagger,
  ner_chunker,
  dependency_parser,
  re_ner_chunk_filter,
  re_model
))

val data = Seq("MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia").toDF("text")
val result = trained_pipeline.fit(data).transform(data)

// Show results
//
// result.selectExpr("explode(relations) as relations")
//  .select(
//    "relations.metadata.chunk1",
//    "relations.metadata.entity1",
//    "relations.metadata.chunk2",
//    "relations.metadata.entity2",
//    "relations.result"
//  )
//  .where("result != 0")
//  .show(truncate=false)
// +------+---------+-------------+---------------------------+------+
// |chunk1|entity1  |chunk2       |entity2                    |result|
// +------+---------+-------------+---------------------------+------+
// |upper |Direction|brain stem   |Internal_organ_or_component|1     |
// |left  |Direction|cerebellum   |Internal_organ_or_component|1     |
// |right |Direction|basil ganglia|Internal_organ_or_component|1     |
// +------+---------+-------------+---------------------------+------+
//
{%- endcapture -%}

{%- capture api_link -%}
[RelationExtractionDLModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/re/RelationExtractionDLModel)
{%- endcapture -%}

{% include templates/licensed_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link%}