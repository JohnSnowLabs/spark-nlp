package sparknlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

trait Annotator extends Transformer with DefaultParamsWritable {

  val aType: String

  def annotate(document: Document, annos: Seq[Annotation]): Seq[Annotation]

  val requiredAnnotationTypes: Seq[String] = Seq()

  def annotateRow(docRow: Row, annos: Seq[Seq[Row]]): Seq[Annotation] = {
    annotate(
      Document(docRow),
      annos.flatten.map(Annotation(_)))
  }

  protected def outputDataType: DataType = ArrayType(Annotation.AnnotationDataType)

  override val uid: String = aType

  val documentCol: Param[String] =
    new Param(this, "documentCol", "the input document column")

  def setDocumentCol(value: String): this.type = set(documentCol, value)

  def getDocumentCol: String = $(documentCol)

  val inputAnnotationCols: Param[Array[String]] =
    new Param(this, "inputAnnotationCols", "the input annotation columns")

  def setInputAnnotationCols(value: Array[String]): this.type = set(inputAnnotationCols, value)

  def getInputAnnotationCols: Array[String] = $(inputAnnotationCols)

  setDefault(inputAnnotationCols, Array[String]())

  def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val transformUDF = udf(this.annotateRow _, outputDataType)
    val annoCols = $(inputAnnotationCols).map(dataset(_))
    dataset.withColumn(aType, transformUDF(dataset($(documentCol)), array(annoCols: _*)))
  }

  override def transformSchema(schema: StructType): StructType = {
    require(schema($(documentCol)).dataType == Document.DocumentDataType,
      s"documentCol [${$(documentCol)}] must be a document column")
    $(inputAnnotationCols).foreach {
      annoCol =>
        require(schema(annoCol).dataType == ArrayType(Annotation.AnnotationDataType),
          s"annotation column [$annoCol] must be an annotation column")
    }
    if (schema.fieldNames.contains(aType)) {
      throw new IllegalArgumentException(s"Output column $aType already exists.")
    }

    val outputFields = schema.fields :+
      StructField(aType, outputDataType, nullable = false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}