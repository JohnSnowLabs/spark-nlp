package com.johnsnowlabs.nlp.finisher

import com.johnsnowlabs.nlp.AnnotatorType
import com.johnsnowlabs.nlp.util.FinisherUtil
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{col, element_at}
import org.apache.spark.sql.types.{ArrayType, FloatType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

case class DocumentSimilarityRankerFinisher (override val uid: String)
  extends Transformer
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("document_similarity_ranker_finisher"))

  val LSH_ID_COL_NAME = "lshId"

  val LSH_NEIGHBORS_COL_NAME = "lshNeighbors"

  /** Name of input annotation cols containing embeddings
   *
   * @group param
   */
  val inputCols: StringArrayParam =
    new StringArrayParam(this, "inputCols", "Name of input annotation cols containing similar documents")

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
    new StringArrayParam(this, "outputCols", "Name of DocumentSimilarityRankerFinisher output cols")

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

  override def transform(dataset: Dataset[_]): DataFrame = {
    val finisherBaseColName = "finished_doc_similarity_ranker"
    dataset
      .withColumn(
        s"${finisherBaseColName}_id",
        element_at(col(s"${AnnotatorType.DOC_SIMILARITY_RANKINGS}.metadata"), 1)
          .getItem(LSH_ID_COL_NAME)
      )
      .withColumn(s"${finisherBaseColName}_neighbors",
        element_at(col(s"${AnnotatorType.DOC_SIMILARITY_RANKINGS}.metadata"), 1)
          .getItem(LSH_NEIGHBORS_COL_NAME)
      )
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val documentSimilarityRankerAnnotators = Seq(AnnotatorType.DOC_SIMILARITY_RANKINGS)

    getInputCols.foreach { annotationColumn =>
      FinisherUtil.checkIfInputColsExist(getInputCols, schema)
      FinisherUtil.checkIfAnnotationColumnIsSparkNLPAnnotation(schema, annotationColumn)

      /** Check if the annotationColumn has DocumentSimilarityRanker. It must be
       * annotators: DocumentSimilarityRanker
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
