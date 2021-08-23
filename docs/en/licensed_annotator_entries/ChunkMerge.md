{%- capture title -%}
ChunkMerge
{%- endcapture -%}

{%- capture model_description -%}
Merges entities coming from different CHUNK annotations
{%- endcapture -%}

{%- capture model_input_anno -%}
CHUNK, CHUNK
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_api_link -%}
[ChunkMergeModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/merge/ChunkMergeModel)
{%- endcapture -%}

{%- capture approach_description -%}
Merges two chunk columns coming from two annotators(NER, ContextualParser or any other annotator producing
chunks). The merger of the two chunk columns is made by selecting one chunk from one of the columns according
to certain criteria.
The decision on which chunk to select is made according to the chunk indices in the source document.
(chunks with longer lengths and highest information will be kept from each source)
Labels can be changed by setReplaceDictResource.
{%- endcapture -%}

{%- capture approach_input_anno -%}
CHUNK, CHUNK
{%- endcapture -%}

{%- capture approach_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
# Define a pipeline with 2 different NER models with a ChunkMergeApproach at the end
data = spark.createDataFrame([["A 63-year-old man presents to the hospital ..."]]).toDF("text")
pipeline = Pipeline(stages=[
 DocumentAssembler().setInputCol("text").setOutputCol("document"),
 SentenceDetector().setInputCols(["document"]).setOutputCol("sentence"),
 Tokenizer().setInputCols(["sentence"]).setOutputCol("token"),
  WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models").setOutputCol("embs"),
  MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embs"]).setOutputCol("jsl_ner"),
 NerConverter().setInputCols(["sentence", "token", "jsl_ner"]).setOutputCol("jsl_ner_chunk"),
  MedicalNerModel.pretrained("ner_bionlp", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embs"]).setOutputCol("bionlp_ner"),
 NerConverter().setInputCols(["sentence", "token", "bionlp_ner"]) \
    .setOutputCol("bionlp_ner_chunk"),
 ChunkMergeApproach().setInputCols(["jsl_ner_chunk", "bionlp_ner_chunk"]).setOutputCol("merged_chunk")
])

# Show results
result = pipeline.fit(data).transform(data).cache()
result.selectExpr("explode(merged_chunk) as a") \
  .selectExpr("a.begin","a.end","a.result as chunk","a.metadata.entity as entity") \
  .show(5, False)
+-----+---+-----------+---------+
|begin|end|chunk      |entity   |
+-----+---+-----------+---------+
|5    |15 |63-year-old|Age      |
|17   |19 |man        |Gender   |
|64   |72 |recurrent  |Modifier |
|98   |107|cellulitis |Diagnosis|
|110  |119|pneumonias |Diagnosis|
+-----+---+-----------+---------+

{%- endcapture -%}

{%- capture approach_scala_example -%}
// Define a pipeline with 2 different NER models with a ChunkMergeApproach at the end
val data = Seq(("A 63-year-old man presents to the hospital ...")).toDF("text")
val pipeline = new Pipeline().setStages(Array(
  new DocumentAssembler().setInputCol("text").setOutputCol("document"),
  new SentenceDetector().setInputCols("document").setOutputCol("sentence"),
  new Tokenizer().setInputCols("sentence").setOutputCol("token"),
  WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models").setOutputCol("embs"),
  MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models")
    .setInputCols("sentence", "token", "embs").setOutputCol("jsl_ner"),
  new NerConverter().setInputCols("sentence", "token", "jsl_ner").setOutputCol("jsl_ner_chunk"),
  MedicalNerModel.pretrained("ner_bionlp", "en", "clinical/models")
    .setInputCols("sentence", "token", "embs").setOutputCol("bionlp_ner"),
  new NerConverter().setInputCols("sentence", "token", "bionlp_ner")
    .setOutputCol("bionlp_ner_chunk"),
  new ChunkMergeApproach().setInputCols("jsl_ner_chunk", "bionlp_ner_chunk").setOutputCol("merged_chunk")
))

// Show results
val result = pipeline.fit(data).transform(data).cache()
result.selectExpr("explode(merged_chunk) as a")
  .selectExpr("a.begin","a.end","a.result as chunk","a.metadata.entity as entity")
  .show(5, false)
+-----+---+-----------+---------+
|begin|end|chunk      |entity   |
+-----+---+-----------+---------+
|5    |15 |63-year-old|Age      |
|17   |19 |man        |Gender   |
|64   |72 |recurrent  |Modifier |
|98   |107|cellulitis |Diagnosis|
|110  |119|pneumonias |Diagnosis|
+-----+---+-----------+---------+

{%- endcapture -%}

{%- capture approach_api_link -%}
[ChunkMergeApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/merge/ChunkMergeApproach)
{%- endcapture -%}


{% include templates/licensed_approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_api_link=model_api_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_api_link=approach_api_link
%}
