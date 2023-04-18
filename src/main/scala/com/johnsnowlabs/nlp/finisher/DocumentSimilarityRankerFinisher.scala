package com.johnsnowlabs.nlp.finisher

import com.johnsnowlabs.nlp.AnnotatorType
import com.johnsnowlabs.nlp.util.FinisherUtil
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

case class DocumentSimilarityRankerFinisher(override val uid: String)
    extends Transformer
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("DOCUMENT_SIMILARITY_RANKER_FINISHER"))

  val LSH_ID_COL_NAME = "lshId"

  val LSH_NEIGHBORS_COL_NAME = "lshNeighbors"

  val FINISHED_DOC_SIM_RANKER_ID_DEFAULT = "finished_doc_similarity_rankings_id"

  val FINISHED_DOC_SIM_RANKER_NEIGHBORS_DEFAULT = "finished_doc_similarity_rankings_neighbors"

  /** Name of input annotation cols containing embeddings
    *
    * @group param
    */
  val inputCols: StringArrayParam =
    new StringArrayParam(
      this,
      "inputCols",
      "Name of input annotation cols containing similar documents")

  /** Name of input annotation cols containing similar documents
    *
    * @group setParam
    */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /** Name of input annotation cols containing similar documents
    *
    * @group setParam
    */
  def setInputCols(value: String*): this.type = setInputCols(value.toArray)

  /** Name of DocumentSimilarityRankerFinisher output cols
    *
    * @group getParam
    */
  def getInputCols: Array[String] = $(inputCols)

  /** Name of DocumentSimilarityRankerFinisher output cols
    *
    * @group param
    */
  val outputCols: StringArrayParam =
    new StringArrayParam(
      this,
      "outputCols",
      "Name of DocumentSimilarityRankerFinisher output cols")

  /** Name of DocumentSimilarityRankerFinisher output cols
    *
    * @group setParam
    */
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  /** Name of DocumentSimilarityRankerFinisher output cols
    *
    * @group setParam
    */
  def setOutputCols(value: String*): this.type = setOutputCols(value.toArray)

  /** Name of input annotation cols containing embeddings
    *
    * @group getParam
    */
  def getOutputCols: Array[String] = get(outputCols).getOrElse(getInputCols.map("finished_" + _))

  val extractNearestNeighbor: BooleanParam =
    new BooleanParam(
      this,
      "extractNearestNeighbor",
      doc = "Extract the best neighbors with distance")

  /** Set flag to extract best neighbor with distance
    *
    * @group setParam
    */
  def setExtractNearestNeighbor(value: Boolean): this.type = set(extractNearestNeighbor, value)

  /** Name of input annotation cols containing embeddings
    *
    * @group getParam
    */
  def getExtractNearestNeighbor: Boolean = $(extractNearestNeighbor)

  setDefault(extractNearestNeighbor -> false)

  override def transform(dataset: Dataset[_]): DataFrame = {

    require(
      getOutputCols.length == 1 || getOutputCols.length == 2,
      "Output column array should have length 1 (default case) or 2 when value id and neighbors are assigned.")

    val (idColName, neighborsColName) =
      getOutputCols.length match {
        case 1 => (FINISHED_DOC_SIM_RANKER_ID_DEFAULT, FINISHED_DOC_SIM_RANKER_NEIGHBORS_DEFAULT)
        case 2 => (getOutputCols(0), getOutputCols(1))
      }

    val transformed = dataset
      .withColumn(
        idColName,
        element_at(col(s"${AnnotatorType.DOC_SIMILARITY_RANKINGS}.metadata"), 1)
          .getItem(LSH_ID_COL_NAME)
          .cast("int"))
      .withColumn(
        neighborsColName,
        element_at(col(s"${AnnotatorType.DOC_SIMILARITY_RANKINGS}.metadata"), 1)
          .getItem(LSH_NEIGHBORS_COL_NAME))

    val formatted = transformed
      .withColumn(
        s"no_squared_$neighborsColName",
        regexp_replace(col(neighborsColName), "[\\[\\]]", ""))
      .withColumn(
        s"tuple_extract_$neighborsColName",
        regexp_extract(col(s"no_squared_$neighborsColName"), "\\((.*?)\\)", 0))
      .withColumn(
        s"no_rounded_$neighborsColName",
        regexp_replace(col(s"tuple_extract_$neighborsColName"), "[\\(\\)]", ""))

    val result =
      if (getExtractNearestNeighbor)
        formatted
          .withColumn(
            s"split_$neighborsColName",
            split(col(s"no_rounded_$neighborsColName"), ","))
          .withColumn(
            "nearest_neighbor_id",
            element_at(col(s"split_$neighborsColName"), 1).cast(IntegerType))
          .withColumn("nearest_neighbor_distance", element_at(col(s"split_$neighborsColName"), 2))
      else
        formatted

    result.drop(
      s"no_squared_$neighborsColName",
      s"tuple_extract_$neighborsColName",
      s"no_rounded_$neighborsColName",
      s"split_$neighborsColName")
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val documentSimilarityRankerAnnotators = Seq(AnnotatorType.DOC_SIMILARITY_RANKINGS)

    getInputCols.foreach { annotationColumn =>
      FinisherUtil.checkIfInputColsExist(getInputCols, schema)
      FinisherUtil.checkIfAnnotationColumnIsSparkNLPAnnotation(schema, annotationColumn)

      /** Check if the annotationColumn has DocumentSimilarityRanker. It must be annotators:
        * DocumentSimilarityRanker
        */
      require(
        documentSimilarityRankerAnnotators.contains(
          schema(annotationColumn).metadata.getString("annotatorType")),
        s"column [$annotationColumn] must be of type DocumentSimilarityRanker")
    }

    val outputFields = schema.fields

    StructType(outputFields)
  }
}

object DocumentSimilarityRankerFinisher
    extends DefaultParamsReadable[DocumentSimilarityRankerFinisher]
