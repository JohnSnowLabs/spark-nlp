{%- capture title -%}
DocumentSimilarityRanker
{%- endcapture -%}

{%- capture model_description -%}
Instantiated model of the DocumentSimilarityRankerApproach. For usage and examples see the
documentation of the main class.
{%- endcapture -%}

{%- capture model_input_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture model_output_anno -%}
DOC_SIMILARITY_RANKINGS
{%- endcapture -%}

{%- capture model_api_link -%}
[DocumentSimilarityRankerModel](/api/com/johnsnowlabs/nlp/annotators/similarity/DocumentSimilarityRankerModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[DocumentSimilarityRankerModel](TODO: implement new for new scheme)
{%- endcapture -%}

{%- capture model_source_link -%}
[DocumentSimilarityRankerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/similarity/DocumentSimilarityRankerModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Annotator that uses LSH techniques present in Spark ML lib to execute approximate nearest
neighbors search on top of sentence embeddings.

It aims to capture the semantic meaning of a document in a dense, continuous vector space and
return it to the ranker search.

For instantiated/pretrained models, see DocumentSimilarityRankerModel.

For extended examples of usage, see the jupyter notebook
[Document Similarity Ranker for Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/text-similarity/doc-sim-ranker/test_doc_sim_ranker.ipynb).
{%- endcapture -%}

{%- capture approach_input_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_output_anno -%}
DOC_SIMILARITY_RANKINGS
{%- endcapture -%}

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from sparknlp.annotator.similarity.document_similarity_ranker import *

document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
sentence_embeddings = E5Embeddings.pretrained() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence_embeddings")
document_similarity_ranker = DocumentSimilarityRankerApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("doc_similarity_rankings") \
            .setSimilarityMethod("brp") \
            .setNumberOfNeighbours(1) \
            .setBucketLength(2.0) \
            .setNumHashTables(3) \
            .setVisibleDistances(True) \
            .setIdentityRanking(False)
document_similarity_ranker_finisher = DocumentSimilarityRankerFinisher() \
        .setInputCols("doc_similarity_rankings") \
        .setOutputCols(
            "finished_doc_similarity_rankings_id",
            "finished_doc_similarity_rankings_neighbors") \
        .setExtractNearestNeighbor(True)
pipeline = Pipeline(stages=[
            document_assembler,
            sentence_embeddings,
            document_similarity_ranker,
            document_similarity_ranker_finisher
        ])
docSimRankerPipeline = pipeline.fit(data).transform(data)

(
    docSimRankerPipeline
        .select(
               "finished_doc_similarity_rankings_id",
               "finished_doc_similarity_rankings_neighbors"
        ).show(10, False)
)
+-----------------------------------+------------------------------------------+
|finished_doc_similarity_rankings_id|finished_doc_similarity_rankings_neighbors|
+-----------------------------------+------------------------------------------+
|1510101612                         |[(1634839239,0.12448559591306324)]        |
|1634839239                         |[(1510101612,0.12448559591306324)]        |
|-612640902                         |[(1274183715,0.1220122862046063)]         |
|1274183715                         |[(-612640902,0.1220122862046063)]         |
|-1320876223                        |[(1293373212,0.17848855164122393)]        |
|1293373212                         |[(-1320876223,0.17848855164122393)]       |
|-1548374770                        |[(-1719102856,0.23297156732534166)]       |
|-1719102856                        |[(-1548374770,0.23297156732534166)]       |
+-----------------------------------+------------------------------------------+

{%- endcapture -%}

{%- capture approach_scala_example -%}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.similarity.DocumentSimilarityRankerApproach
import com.johnsnowlabs.nlp.finisher.DocumentSimilarityRankerFinisher
import org.apache.spark.ml.Pipeline

import spark.implicits._

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceEmbeddings = RoBertaSentenceEmbeddings
  .pretrained()
  .setInputCols("document")
  .setOutputCol("sentence_embeddings")

val documentSimilarityRanker = new DocumentSimilarityRankerApproach()
  .setInputCols("sentence_embeddings")
  .setOutputCol("doc_similarity_rankings")
  .setSimilarityMethod("brp")
  .setNumberOfNeighbours(1)
  .setBucketLength(2.0)
  .setNumHashTables(3)
  .setVisibleDistances(true)
  .setIdentityRanking(false)

val documentSimilarityRankerFinisher = new DocumentSimilarityRankerFinisher()
  .setInputCols("doc_similarity_rankings")
  .setOutputCols(
    "finished_doc_similarity_rankings_id",
    "finished_doc_similarity_rankings_neighbors")
  .setExtractNearestNeighbor(true)

// Let's use a dataset where we can visually control similarity
// Documents are coupled, as 1-2, 3-4, 5-6, 7-8 and they were create to be similar on purpose
val data = Seq(
  "First document, this is my first sentence. This is my second sentence.",
  "Second document, this is my second sentence. This is my second sentence.",
  "Third document, climate change is arguably one of the most pressing problems of our time.",
  "Fourth document, climate change is definitely one of the most pressing problems of our time.",
  "Fifth document, Florence in Italy, is among the most beautiful cities in Europe.",
  "Sixth document, Florence in Italy, is a very beautiful city in Europe like Lyon in France.",
  "Seventh document, the French Riviera is the Mediterranean coastline of the southeast corner of France.",
  "Eighth document, the warmest place in France is the French Riviera coast in Southern France.")
  .toDF("text")

val pipeline = new Pipeline().setStages(
  Array(
    documentAssembler,
    sentenceEmbeddings,
    documentSimilarityRanker,
    documentSimilarityRankerFinisher))

val result = pipeline.fit(data).transform(data)

result
  .select("finished_doc_similarity_rankings_id", "finished_doc_similarity_rankings_neighbors")
  .show(10, truncate = false)
+-----------------------------------+------------------------------------------+
|finished_doc_similarity_rankings_id|finished_doc_similarity_rankings_neighbors|
+-----------------------------------+------------------------------------------+
|1510101612                         |[(1634839239,0.12448559591306324)]        |
|1634839239                         |[(1510101612,0.12448559591306324)]        |
|-612640902                         |[(1274183715,0.1220122862046063)]         |
|1274183715                         |[(-612640902,0.1220122862046063)]         |
|-1320876223                        |[(1293373212,0.17848855164122393)]        |
|1293373212                         |[(-1320876223,0.17848855164122393)]       |
|-1548374770                        |[(-1719102856,0.23297156732534166)]       |
|-1719102856                        |[(-1548374770,0.23297156732534166)]       |
+-----------------------------------+------------------------------------------+

{%- endcapture -%}

{%- capture approach_api_link -%}
[DocumentSimilarityRankerApproach](/api/com/johnsnowlabs/nlp/annotators/similarity/DocumentSimilarityRankerApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[DocumentSimilarityRankerApproach](/api/python/reference/autosummary/sparknlp/annotator/similarity/document_similarity_ranker/index.html#sparknlp.annotator.similarity.document_similarity_ranker.DocumentSimilarityRankerApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[DocumentSimilarityRankerApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/similarity/DocumentSimilarityRankerApproach.scala)
{%- endcapture -%}


{% include templates/approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_api_link=model_api_link
model_python_api_link=model_python_api_link
model_source_link=model_source_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_api_link=approach_api_link
approach_python_api_link=approach_python_api_link
approach_source_link=approach_source_link
%}
