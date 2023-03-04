package com.johnsnowlabs.nlp.annotators.similarity

import com.johnsnowlabs.nlp.AnnotatorType.{DOC_SIMILARITY_RANKINGS, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.{AnnotatorApproach, HasEnableCachingProperties}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, BucketedRandomProjectionLSHModel, VectorAssembler}
import org.apache.spark.ml.functions.array_to_vector
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions.{col, expr, flatten, hash, monotonically_increasing_id, row_number, udf}

import scala.util.hashing.MurmurHash3

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

  /** The similarity method used to calculate the neighbours.
   * (Default: `"brp"`, Bucketed Random Projection for Euclidean Distance)
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
    """The number of neighbours the model will return (Default:`"10"`)""")

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

  def getNumHashTables: Int = $(numHashTables)

  setDefault(
    inputCols -> Array(SENTENCE_EMBEDDINGS),
    outputCol -> DOC_SIMILARITY_RANKINGS,
    similarityMethod -> "brp",
    numberOfNeighbours -> 10,
    bucketLength -> 2.0,
    numHashTables -> 3
  )

  val LSH_INPUT_COL_NAME = "features"

  val LSH_OUTPUT_COL_NAME = "hashes"

  val INDEX_COL_NAME = "index"

  def getANN(model: BucketedRandomProjectionLSHModel, query: (String, DenseVector), similarityDataset: DataFrame) = {
    query match {
      case (index, key) =>
        val similarRankedDocs = model.approxNearestNeighbors(similarityDataset, key, getNumberOfNeighbours)
        val neighborsStr = similarRankedDocs.select(INDEX_COL_NAME).collect().map(_.getString(0)).mkString("|")
        index.concat("=>").concat(neighborsStr)
    }
  }

  val INPUT_EMBEDDINGS = "sentence_embeddings.embeddings"

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DocumentSimilarityRankerModel = {
    val lsh = $(similarityMethod) match {
      case "brp" => new BucketedRandomProjectionLSH()
        .setBucketLength($(bucketLength))
        .setNumHashTables($(numHashTables))
        .setInputCol(LSH_INPUT_COL_NAME)
        .setOutputCol(LSH_OUTPUT_COL_NAME)
      case _ => throw new IllegalArgumentException(s"${$(similarityMethod)} is not a valid value.")
    }

    val embeddingsDataset = dataset.withColumn(LSH_INPUT_COL_NAME, col(INPUT_EMBEDDINGS))

    val similarityDataset: DataFrame = embeddingsDataset
      .withColumn(s"$LSH_INPUT_COL_NAME", flatten(col(s"$LSH_INPUT_COL_NAME")))
      .withColumn(s"$LSH_INPUT_COL_NAME", array_to_vector(col(s"$LSH_INPUT_COL_NAME")))

    val model = lsh.fit(similarityDataset)

    val mh3UDF = udf{
      (s: String) => MurmurHash3.stringHash(s, MurmurHash3.stringSeed).toString
    }

    val similarityDatasetWithIndex = similarityDataset.withColumn(INDEX_COL_NAME, mh3UDF(col("text")))

    val indexedVectorTuples = similarityDatasetWithIndex
      .select(INDEX_COL_NAME, LSH_INPUT_COL_NAME)
      .rdd
      .map(x => (x.getAs(INDEX_COL_NAME), x.getAs(LSH_INPUT_COL_NAME))).collect()

    val similarityMappings: Array[String] = indexedVectorTuples
      .map(query => getANN(model, query, similarityDatasetWithIndex))

    new DocumentSimilarityRankerModel()
      .setSimilarityMappings(Map("similarityMappings" -> similarityMappings))
  }
}
