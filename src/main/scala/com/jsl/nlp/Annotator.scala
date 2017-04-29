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
  type DataContent = Row

  /**
    * This is the annotation type
    */
  val aType: String

  /**
    * This parameter tells the annotator the column that contains the document
    */
  val documentCol: Param[String] =
    new Param(this, "documentCol", "the input document column")

  /**
    * This parameter tells the annotator the columns that contain the annotations necessary to run this annotator
    * (empty by default)
    */
  val inputAnnotationCols: Param[Array[String]] =
    new Param(this, "inputAnnotationCols", "the input annotation columns")
  setDefault(inputAnnotationCols, Array[String]())

  /**
    * This is the annotation types that this annotator expects to be present
    */
  val requiredAnnotationTypes: Seq[String] = Seq()

  val outputAnnotationCol: Param[String] =
    new Param(this, "outputAnnotationCol", "the output annotation column")

  override val uid: String = aType

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
    * @return
    */
  def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation]

  /**
    * Wraps annotate to happen inside SparkSQL user defined functions
    * @return
    */
  def dfAnnotate: UserDefinedFunction = udf {
    (docProperties: DataContent, aProperties: Seq[DataContent]) =>
      annotate(Document(docProperties), aProperties.map(Annotation(_)))
  }

  def setDocumentCol(value: String): this.type = set(documentCol, value)

  def getDocumentCol: String = $(documentCol)

  def setInputAnnotationCols(value: Array[String]): this.type = set(inputAnnotationCols, value)

  def getInputAnnotationCols: Array[String] = $(inputAnnotationCols)

  /**
    * Dummy code --> Mimic transform in dataframe api layer
    * @param dataFrame Until User Defined Types arrive, this will be Dataset[Row]
    * @return
    */
  override def transform(dataFrame: Dataset[_]): DataFrame = {
    dataFrame.withColumn(
      $(documentCol),
      dfAnnotate(
        dataFrame.col($(documentCol)),
        array($(inputAnnotationCols).map(c => dataFrame.col(c)):_*)
      )
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

    val outputFields = schema.fields :+
      StructField(aType, outputDataType, nullable = false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  protected def outputDataType: DataType = ArrayType(Annotation.AnnotationDataType)
}