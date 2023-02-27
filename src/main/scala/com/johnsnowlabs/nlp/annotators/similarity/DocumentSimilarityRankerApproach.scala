package com.johnsnowlabs.nlp.annotators.similarity

import com.johnsnowlabs.nlp.AnnotatorType.{DOC_SIMILARITY_RANKINGS, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.{AnnotatorApproach, HasEnableCachingProperties}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.sql.Dataset

class DocumentSimilarityRankerApproach(override val uid: String)
extends AnnotatorApproach[DocumentSimilarityRankerModel]
  with HasStorageRef
  with HasEnableCachingProperties {
  override val description: AnnotatorType = ???

  override val outputAnnotatorType: AnnotatorType = ???
  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
   * type
   */
  override val inputAnnotatorTypes: Array[AnnotatorType] = ???

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

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DocumentSimilarityRankerModel = ???

}
