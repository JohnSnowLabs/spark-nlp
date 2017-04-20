package sparknlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.sql._
import org.apache.spark.sql.types._

/**
  * This trait is for implementing logic that finds segments of text.
  */
trait Annotator extends Transformer {

  /**
    * This is the annotation type
    */
  val aType: String

  /**
    * This is the annotation types that this annotator expects to be present
    */
  val requiredAnnotationTypes: Seq[String] = Seq()

  /**
    * This takes a [[DataFrame]] and checks to see if all the required annotation types are present.
    * @param dataFrame The dataframe to be validated
    * @return True if all the required types are present, else false
    */
  def validate(dataFrame: DataFrame): Boolean = requiredAnnotationTypes.forall{
    requiredAnnotationType =>
      dataFrame.schema.exists {
        field =>
          field.metadata.contains("annotationType") &&
            field.metadata.getString("annotationType") == requiredAnnotationType
      }
  }

  /**
    * This takes a document and annotations and produces new annotations of this annotator's annotation type
    * @param document
    * @param annos
    * @return
    */
  def annotate(document: Document, annos: Seq[Annotation]): Seq[Annotation]

  /**
    * This takes a document ([[Row]]) and the annotations
    * @param docRow
    * @param annos
    * @return
    */
  def annotateRow(docRow: Row, annos: Seq[Seq[Row]]): Seq[Annotation] = {
    annotate(
      Document(docRow),
      annos.flatten.map(Annotation(_)))
  }

  /**
    * This parameter tells the annotator the column that contains the document
    */
  val documentCol: Param[String] =
    new Param(this, "documentCol", "the input document column")

  def setDocumentCol(value: String): this.type = set(documentCol, value)

  def getDocumentCol: String = $(documentCol)

  /**
    * This parameter tells the annotator the columns that contain the annotations necessary to run this annotator
    * (empty by default)
    */
  val inputAnnotationCols: Param[Array[String]] =
    new Param(this, "inputAnnotationCols", "the input annotation columns")

  def setInputAnnotationCols(value: Array[String]): this.type = set(inputAnnotationCols, value)

  def getInputAnnotationCols: Array[String] = $(inputAnnotationCols)

  setDefault(inputAnnotationCols, Array[String]())

  def transform(dataset: DataFrame): DataFrame = {
    val schema = dataset.columns.zipWithIndex.toMap
    val sqlc = dataset.sqlContext
    sqlc.createDataFrame(dataset.rdd.map {
      row =>
        val docRow = row.getAs[Row](schema($(documentCol)))
        val prevAnnotations = $(inputAnnotationCols).map(c => row.getSeq[Row](schema(c)))
        val annotations = annotateRow(docRow, prevAnnotations).toList
        Row(row.toSeq :+ annotations: _*)
    }, dataset.schema.add(aType, ArrayType(Annotation.AnnotationDataType)))
  }

  override def transformSchema(schema: StructType): StructType = {
    require(schema($(documentCol)).dataType == Document.DocumentDataType,
      s"documentCol [${$(documentCol)}] must be a document column")
    $(inputAnnotationCols).foreach {
      annoCol =>
        require(schema(annoCol).dataType == ArrayType(Annotation.AnnotationDataType),
          s"annotation column [$annoCol] must be an annotation column, found [${schema(annoCol).dataType}]")
    }
    if (schema.fieldNames.contains(aType)) {
      throw new IllegalArgumentException(s"Output column $aType already exists.")
    }

    val outputFields = schema.fields :+
      StructField(aType, outputDataType, nullable = false)
    StructType(outputFields)
  }

  override val uid: String = aType

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  protected def outputDataType: DataType = ArrayType(Annotation.AnnotationDataType)
}