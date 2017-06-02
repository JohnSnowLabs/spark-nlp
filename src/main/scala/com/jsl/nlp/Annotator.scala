package com.jsl.nlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.{array, udf}

/**
  * This trait is for implementing logic that finds segments of text.
  */
trait Annotator extends Transformer with DefaultParamsWritable {

  /**
    * This is an internal type to show Rows as a relevant StructType
    * Should be deleted once Spark releases UserDefinedTypes to @developerAPI
    */
  type DocumentContent = Row
  type AnnotationContent = Seq[Row]

  /**
    * This is the annotation type
    */
  protected val aType: String

  /**
    * This is the annotation types that this annotator expects to be present
    */
  protected val requiredAnnotationTypes: Array[String]

  /**
    * This parameter tells the annotator the column that contains the document
    */
  private val documentCol: Param[String] =
    new Param(this, "document column", "the input document column")

  /**
    * This parameter tells the annotator the columns that contain the annotations necessary to run this annotator
    * (empty by default)
    */
  private val inputAnnotationCols: Param[Array[String]] =
    new Param(this, "inputAnnotationCols", "the input annotation columns")

  private val outputAnnotationCol: Param[String] =
    new Param(this, "outputAnnotationCol", "the output annotation column")

  override val uid: String = aType

  /**
    * This takes a document and annotations and produces new annotations of this annotator's annotation type
    * @return
    */
  protected def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation]

  /**
    * This takes a [[DataFrame]] and checks to see if all the required annotation types are present.
    * @param dataFrame The dataframe to be validated
    * @return True if all the required types are present, else false
    */
  private def validate(dataFrame: Dataset[_]): Boolean = requiredAnnotationTypes.forall {
    requiredAnnotationType =>
      dataFrame.schema.exists {
        field =>
          field.metadata.contains("annotationType") &&
            field.metadata.getString("annotationType") == requiredAnnotationType
      }
  }

  /**
    * Wraps annotate to happen inside SparkSQL user defined functions
    * @return
    */
  private def dfAnnotate: UserDefinedFunction = udf {
    (docProperties: DocumentContent, aProperties: Seq[AnnotationContent]) =>
      annotate(Document(docProperties), aProperties.flatMap(_.map(Annotation(_))))
  }

  private def outputDataType: DataType = ArrayType(Annotation.AnnotationDataType)

  def setDocumentCol(value: String): this.type = set(documentCol, value)

  def getDocumentCol: String = $(documentCol)

  def setInputAnnotationCols(value: Array[String]): this.type = set(inputAnnotationCols, value)

  def getInputAnnotationCols: Array[String] = get(inputAnnotationCols).getOrElse(requiredAnnotationTypes)

  def setOutputAnnotationCol(value: String): this.type = set(outputAnnotationCol, value)

  def getOutputAnnotationCol: String = get(outputAnnotationCol).getOrElse(aType)

  /**
    * Dummy code --> Mimic transform in dataframe api layer
    * @param dataFrame Until User Defined Types arrive, this will be Dataset[Row]
    * @return
    */
  override def transform(dataFrame: Dataset[_]): DataFrame = {
    require(validate(dataFrame), s"DataFrame has unmet requirements: ${requiredAnnotationTypes.mkString(", ")}")
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotationType", aType)
    dataFrame.withColumn(
      getOutputAnnotationCol,
      dfAnnotate(
        dataFrame.col($(documentCol)),
        array(getInputAnnotationCols.map(c => dataFrame.col(c)):_*)
      ).as(getOutputAnnotationCol, metadataBuilder.build)
    )
  }

  override def transformSchema(schema: StructType): StructType = {
    require(schema($(documentCol)).dataType == Document.DocumentDataType,
      s"documentCol [${$(documentCol)}] must be a document column")
    $(inputAnnotationCols).foreach {
      annotationColumn =>
        require(schema(annotationColumn).dataType == ArrayType(Annotation.AnnotationDataType),
          s"annotation column [$annotationColumn] must be an annotation column, found [${schema(annotationColumn).dataType}]")
    }
    if (schema.fieldNames.contains(aType)) {
      throw new IllegalArgumentException(s"Output column $aType already exists.")
    }
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    requiredAnnotationTypes.foreach{requiredType => metadataBuilder.putString("annotationType", requiredType)}
    val outputFields = schema.fields :+
      StructField(aType, outputDataType, nullable = false, metadataBuilder.build)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}