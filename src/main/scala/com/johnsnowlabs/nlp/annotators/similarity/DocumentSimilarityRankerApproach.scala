package com.johnsnowlabs.nlp.annotators.similarity

import com.johnsnowlabs.nlp.AnnotatorType.{DOC_SIMILARITY_RANKINGS, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.{AnnotatorApproach, HasEnableCachingProperties}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, BucketedRandomProjectionLSHModel}
import org.apache.spark.ml.functions.array_to_vector
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{col, flatten, udf}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.util.hashing.MurmurHash3

sealed trait NeighborAnnotation {
  def neighbors: Array[_]
}

case class IndexedNeighbors(neighbors: Array[Int]) extends NeighborAnnotation

case class IndexedNeighborsWithDistance(neighbors: Array[(Int, Double)])
    extends NeighborAnnotation

case class NeighborsResultSet(result: (Int, NeighborAnnotation))

class DocumentSimilarityRankerApproach(override val uid: String)
    extends AnnotatorApproach[DocumentSimilarityRankerModel]
    with HasStorageRef
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

  val INPUT_EMBEDDINGS = "sentence_embeddings.embeddings"

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

  setDefault(
    similarityMethod -> "brp",
    numberOfNeighbours -> 10,
    bucketLength -> 2.0,
    numHashTables -> 3,
    visibleDistances -> false,
    identityRanking -> false)

  def getNeighborsResultSet(
      model: BucketedRandomProjectionLSHModel,
      query: (Int, DenseVector),
      similarityDataset: DataFrame): NeighborsResultSet = {
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
    }
  }

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): DocumentSimilarityRankerModel = {

    val lsh = $(similarityMethod) match {
      case "brp" =>
        new BucketedRandomProjectionLSH()
          .setBucketLength($(bucketLength))
          .setNumHashTables($(numHashTables))
          .setInputCol(LSH_INPUT_COL_NAME)
          .setOutputCol(LSH_OUTPUT_COL_NAME)
      case _ =>
        throw new IllegalArgumentException(s"${$(similarityMethod)} is not a valid value.")
    }

    val embeddingsDataset = dataset.withColumn(LSH_INPUT_COL_NAME, col(INPUT_EMBEDDINGS))

    val similarityDataset: DataFrame = embeddingsDataset
      .withColumn(s"$LSH_INPUT_COL_NAME", flatten(col(s"$LSH_INPUT_COL_NAME")))
      .withColumn(s"$LSH_INPUT_COL_NAME", array_to_vector(col(s"$LSH_INPUT_COL_NAME")))

    val model = lsh.fit(similarityDataset)

    val mh3UDF = udf { (s: String) =>
      MurmurHash3.stringHash(s, MurmurHash3.stringSeed)
    }

    val similarityDatasetWithIndex =
      similarityDataset.withColumn(INDEX_COL_NAME, mh3UDF(col(TEXT)))

    val indexedVectorTuples = similarityDatasetWithIndex
      .select(INDEX_COL_NAME, LSH_INPUT_COL_NAME)
      .rdd
      .map(x => (x.getAs[Int](INDEX_COL_NAME), x.getAs[DenseVector](LSH_INPUT_COL_NAME)))
      .collect()

    val similarityMappings: Map[Int, NeighborAnnotation] = indexedVectorTuples
      .map(query => getNeighborsResultSet(model, query, similarityDatasetWithIndex))
      .map(_.result)
      .toMap

    new DocumentSimilarityRankerModel()
      .setSimilarityMappings(Map("similarityMappings" -> similarityMappings))
  }
}
