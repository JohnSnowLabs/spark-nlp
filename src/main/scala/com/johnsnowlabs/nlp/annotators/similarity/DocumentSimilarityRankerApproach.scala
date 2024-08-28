package com.johnsnowlabs.nlp.annotators.similarity

import com.johnsnowlabs.nlp.AnnotatorType.{DOC_SIMILARITY_RANKINGS, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.similarity.DocumentSimilarityUtil._
import com.johnsnowlabs.nlp.{AnnotatorApproach, HasEnableCachingProperties}
import com.johnsnowlabs.util.spark.SparkUtil.retrieveColumnName
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, MinHashLSH}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset}

sealed trait NeighborAnnotation {
  def neighbors: Array[_]
}

case class IndexedNeighbors(neighbors: Array[Int]) extends NeighborAnnotation

case class IndexedNeighborsWithDistance(neighbors: Array[(Int, Double)])
    extends NeighborAnnotation

case class NeighborsResultSet(result: (Int, NeighborAnnotation))

/** Annotator that uses LSH techniques present in Spark ML lib to execute approximate nearest
  * neighbors search on top of sentence embeddings.
  *
  * It aims to capture the semantic meaning of a document in a dense, continuous vector space and
  * return it to the ranker search.
  *
  * For instantiated/pretrained models, see [[DocumentSimilarityRankerModel]].
  *
  * For extended examples of usage, see the jupyter notebook
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/text-similarity/doc-sim-ranker/test_doc_sim_ranker.ipynb Document Similarity Ranker for Spark NLP]].
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import com.johnsnowlabs.nlp.annotators.similarity.DocumentSimilarityRankerApproach
  * import com.johnsnowlabs.nlp.finisher.DocumentSimilarityRankerFinisher
  * import org.apache.spark.ml.Pipeline
  *
  * import spark.implicits._
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentenceEmbeddings = RoBertaSentenceEmbeddings
  *   .pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("sentence_embeddings")
  *
  * val documentSimilarityRanker = new DocumentSimilarityRankerApproach()
  *   .setInputCols("sentence_embeddings")
  *   .setOutputCol("doc_similarity_rankings")
  *   .setSimilarityMethod("brp")
  *   .setNumberOfNeighbours(1)
  *   .setBucketLength(2.0)
  *   .setNumHashTables(3)
  *   .setVisibleDistances(true)
  *   .setIdentityRanking(false)
  *
  * val documentSimilarityRankerFinisher = new DocumentSimilarityRankerFinisher()
  *   .setInputCols("doc_similarity_rankings")
  *   .setOutputCols(
  *     "finished_doc_similarity_rankings_id",
  *     "finished_doc_similarity_rankings_neighbors")
  *   .setExtractNearestNeighbor(true)
  *
  * // Let's use a dataset where we can visually control similarity
  * // Documents are coupled, as 1-2, 3-4, 5-6, 7-8 and they were create to be similar on purpose
  * val data = Seq(
  *   "First document, this is my first sentence. This is my second sentence.",
  *   "Second document, this is my second sentence. This is my second sentence.",
  *   "Third document, climate change is arguably one of the most pressing problems of our time.",
  *   "Fourth document, climate change is definitely one of the most pressing problems of our time.",
  *   "Fifth document, Florence in Italy, is among the most beautiful cities in Europe.",
  *   "Sixth document, Florence in Italy, is a very beautiful city in Europe like Lyon in France.",
  *   "Seventh document, the French Riviera is the Mediterranean coastline of the southeast corner of France.",
  *   "Eighth document, the warmest place in France is the French Riviera coast in Southern France.")
  *   .toDF("text")
  *
  * val pipeline = new Pipeline().setStages(
  *   Array(
  *     documentAssembler,
  *     sentenceEmbeddings,
  *     documentSimilarityRanker,
  *     documentSimilarityRankerFinisher))
  *
  * val result = pipeline.fit(data).transform(data)
  *
  * result
  *   .select("finished_doc_similarity_rankings_id", "finished_doc_similarity_rankings_neighbors")
  *   .show(10, truncate = false)
  * +-----------------------------------+------------------------------------------+
  * |finished_doc_similarity_rankings_id|finished_doc_similarity_rankings_neighbors|
  * +-----------------------------------+------------------------------------------+
  * |1510101612                         |[(1634839239,0.12448559591306324)]        |
  * |1634839239                         |[(1510101612,0.12448559591306324)]        |
  * |-612640902                         |[(1274183715,0.1220122862046063)]         |
  * |1274183715                         |[(-612640902,0.1220122862046063)]         |
  * |-1320876223                        |[(1293373212,0.17848855164122393)]        |
  * |1293373212                         |[(-1320876223,0.17848855164122393)]       |
  * |-1548374770                        |[(-1719102856,0.23297156732534166)]       |
  * |-1719102856                        |[(-1548374770,0.23297156732534166)]       |
  * +-----------------------------------+------------------------------------------+
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class DocumentSimilarityRankerApproach(override val uid: String)
    extends AnnotatorApproach[DocumentSimilarityRankerModel]
    with HasEnableCachingProperties {

  override val description: AnnotatorType = "LSH based document similarity annotator"

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("DocumentSimilarityRankerApproach"))

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(SENTENCE_EMBEDDINGS)

  override val outputAnnotatorType: AnnotatorType = DOC_SIMILARITY_RANKINGS

  val LSH_INPUT_COL_NAME = "features"

  val LSH_OUTPUT_COL_NAME = "hashes"

  val INDEX_COL_NAME = "index"

  val DISTANCE = "distCol"

  val TEXT = "text"

  /** The similarity method used to calculate the neighbours. (Default: `"brp"`, Bucketed Random
    * Projection for Euclidean Distance)
    *
    * @group param
    */
  val similarityMethod = new Param[String](
    this,
    "similarityMethod",
    """The similarity method used to calculate the neighbours.
      |(Default: `"brp"`, Bucketed Random Projection for Euclidean Distance)
      |""".stripMargin)

  def setSimilarityMethod(value: String): this.type = set(similarityMethod, value)

  def getSimilarityMethod: String = $(similarityMethod)

  /** The number of neighbours the model will return (Default:`"10"`).
    *
    * @group param
    */
  val numberOfNeighbours = new Param[Int](
    this,
    "numberOfNeighbours",
    """The number of neighbours the model will return for each document (Default:`"10"`)""")

  def setNumberOfNeighbours(value: Int): this.type = set(numberOfNeighbours, value)

  def getNumberOfNeighbours: Int = $(numberOfNeighbours)

  val bucketLength = new Param[Double](
    this,
    "bucketLength",
    """The bucket length that controls the average size of hash buckets.
      |A larger bucket length (i.e., fewer buckets) increases the probability of features being hashed
      |to the same bucket (increasing the numbers of true and false positives)
      |""".stripMargin)

  def setBucketLength(value: Double): this.type = set(bucketLength, value)

  def getBucketLength: Double = $(bucketLength)

  val numHashTables = new Param[Int](
    this,
    "numHashTables",
    """number of hash tables, where increasing number of hash tables lowers the false negative rate,
      |and decreasing it improves the running performance.
      |""".stripMargin)

  def setNumHashTables(value: Int): this.type = set(numHashTables, value)

  val visibleDistances = new BooleanParam(
    this,
    "visibleDistances",
    "Whether to set visibleDistances in ranking output (Default: `false`)")

  def setVisibleDistances(value: Boolean): this.type = set(visibleDistances, value)

  def getVisibleDistances: Boolean = $(visibleDistances)

  val identityRanking = new BooleanParam(
    this,
    "identityRanking",
    "Whether to include identity in ranking result set. Useful for debug. (Default: `false`)")

  def setIdentityRanking(value: Boolean): this.type = set(identityRanking, value)

  def getIdentityRanking: Boolean = $(identityRanking)

  val asRetrieverQuery = new Param[String](
    this,
    "asRetrieverQuery",
    "Whether to set the model as retriever RAG with a specific query string. (Default: `empty`)")

  def asRetriever(value: String): this.type = set(asRetrieverQuery, value)

  def getAsRetrieverQuery: String = $(asRetrieverQuery)

  /** Specifies the method used to aggregate multiple sentence embeddings into a single vector
    * representation. Options include 'AVERAGE' (compute the mean of all embeddings), 'FIRST' (use
    * the first embedding only), 'MAX' (compute the element-wise maximum across embeddings)
    *
    * Default AVERAGE
    *
    * @group param
    */
  val aggregationMethod = new Param[String](
    this,
    "aggregationMethod",
    "Specifies the method used to aggregate multiple sentence embeddings into a single vector representation.")

  /** Set the method used to aggregate multiple sentence embeddings into a single vector
    * representation. Options include 'AVERAGE' (compute the mean of all embeddings), 'FIRST' (use
    * the first embedding only), 'MAX' (compute the element-wise maximum across embeddings)
    *
    * Default AVERAGE
    *
    * @group param
    */
  def setAggregationMethod(strategy: String): this.type = {
    strategy.toLowerCase() match {
      case "average" => set(aggregationMethod, "AVERAGE")
      case "first" => set(aggregationMethod, "FIRST")
      case "max" => set(aggregationMethod, "MAX")
      case _ => throw new MatchError("aggregationMethod must be AVERAGE, FIRST or MAX")
    }
  }

  def getAggregationMethod: String = $(aggregationMethod)

  setDefault(
    similarityMethod -> "brp",
    numberOfNeighbours -> 10,
    bucketLength -> 2.0,
    numHashTables -> 3,
    visibleDistances -> false,
    identityRanking -> false,
    asRetrieverQuery -> "",
    aggregationMethod -> "AVERAGE")

  def getNeighborsResultSet(
      query: (Int, Vector),
      similarityDataset: DataFrame): NeighborsResultSet = {

    val lsh = $(similarityMethod) match {
      case "brp" =>
        new BucketedRandomProjectionLSH()
          .setBucketLength($(bucketLength))
          .setNumHashTables($(numHashTables))
          .setInputCol(LSH_INPUT_COL_NAME)
          .setOutputCol(LSH_OUTPUT_COL_NAME)
      case "mh" =>
        new MinHashLSH()
          .setNumHashTables($(numHashTables))
          .setInputCol(LSH_INPUT_COL_NAME)
          .setOutputCol(LSH_OUTPUT_COL_NAME)
      case _ =>
        throw new IllegalArgumentException(s"${$(similarityMethod)} is not a valid value.")
    }

    val model = lsh.fit(similarityDataset)

    query match {
      case (index, queryVector) =>
        val _similarityDataset =
          if (getIdentityRanking) {
            similarityDataset
          } else {
            similarityDataset.where(col("index") =!= index)
          }

        val similarRankedDocs =
          model.approxNearestNeighbors(_similarityDataset, queryVector, getNumberOfNeighbours)

        if (getVisibleDistances) {
          val rankedNeighboursWithDistances = similarRankedDocs
            .select(INDEX_COL_NAME, DISTANCE)
            .collect()
            .map(row => (row.getInt(0), row.getDouble(1)))

          NeighborsResultSet((index, IndexedNeighborsWithDistance(rankedNeighboursWithDistances)))
        } else {
          val rankedNeighbours = similarRankedDocs
            .select(INDEX_COL_NAME)
            .collect()
            .map(_.getInt(0))

          NeighborsResultSet(index, IndexedNeighbors(rankedNeighbours))
        }
      case _ =>
        throw new IllegalArgumentException("asRetrieverQuery is not of type (Int, DenseVector)")
    }
  }

  override def train(
      embeddingsDataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): DocumentSimilarityRankerModel = {

    val inputEmbeddingsColumn =
      s"${retrieveColumnName(embeddingsDataset, SENTENCE_EMBEDDINGS)}.embeddings"

    val similarityDataset: DataFrame = getAggregationMethod match {
      case "AVERAGE" =>
        embeddingsDataset
          .withColumn(s"$LSH_INPUT_COL_NAME", averageAggregation(col(inputEmbeddingsColumn)))
      case "FIRST" =>
        embeddingsDataset
          .withColumn(
            s"$LSH_INPUT_COL_NAME",
            firstEmbeddingAggregation(col(inputEmbeddingsColumn)))
      case "MAX" =>
        embeddingsDataset
          .withColumn(s"$LSH_INPUT_COL_NAME", maxAggregation(col(inputEmbeddingsColumn)))
    }

    val similarityDatasetWithHashIndex =
      similarityDataset.withColumn(INDEX_COL_NAME, mh3UDF(col(TEXT)))

    val indexedVectorTuples = similarityDatasetWithHashIndex
      .select(INDEX_COL_NAME, LSH_INPUT_COL_NAME)
      .rdd
      .map(x => (x.getAs[Int](INDEX_COL_NAME), x.getAs[Vector](LSH_INPUT_COL_NAME)))
      .collect()

    val asRetrieverQuery = getAsRetrieverQuery

    val similarityMappings: Map[Int, NeighborAnnotation] =
      if (asRetrieverQuery.isEmpty)
        indexedVectorTuples
          .map(query => getNeighborsResultSet(query, similarityDatasetWithHashIndex))
          .map(_.result)
          .toMap
      else
        similarityDatasetWithHashIndex
          .where(col("text") === asRetrieverQuery)
          .select(INDEX_COL_NAME, LSH_INPUT_COL_NAME)
          .rdd
          .map(x => (x.getAs[Int](INDEX_COL_NAME), x.getAs[Vector](LSH_INPUT_COL_NAME)))
          .collect()
          .map(asRetrieverQuery =>
            getNeighborsResultSet(asRetrieverQuery, similarityDatasetWithHashIndex))
          .map(_.result)
          .toMap

    new DocumentSimilarityRankerModel()
      .setSimilarityMappings(Map("similarityMappings" -> similarityMappings))
  }
}

/** This is the companion object of [[DocumentSimilarityRankerApproach]]. Please refer to that
  * class for the documentation.
  */
object DocumentSimilarityRankerApproach
    extends DefaultParamsReadable[DocumentSimilarityRankerApproach]
