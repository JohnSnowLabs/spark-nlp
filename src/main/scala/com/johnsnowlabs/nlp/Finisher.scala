package com.johnsnowlabs.nlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types._

class Finisher(override val uid: String)
  extends Transformer
    with DefaultParamsWritable {

  protected val inputCols: StringArrayParam =
    new StringArrayParam(this, "inputCols", "name of input annotation cols")
  protected val outputCols: StringArrayParam =
    new StringArrayParam(this, "outputCols", "name of finisher output cols")
  protected val valueSplitSymbol: Param[String] =
    new Param(this, "valueSplitSymbol", "character separating annotations")
  protected val annotationSplitSymbol: Param[String] =
    new Param(this, "annotationSplitSymbol", "character separating annotations")
  protected val cleanAnnotations: BooleanParam =
    new BooleanParam(this, "cleanAnnotations", "whether to remove annotation columns")
  protected val includeMetadata: BooleanParam =
    new BooleanParam(this, "includeMetadata", "annotation metadata format")
  protected val outputAsArray: BooleanParam =
    new BooleanParam(this, "outputAsArray", "finisher generates an Array with the results instead of string")

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)
  def setInputCols(value: String*): this.type = setInputCols(value.toArray)
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)
  def setOutputCols(value: String*): this.type = setOutputCols(value.toArray)
  def setValueSplitSymbol(value: String): this.type = set(valueSplitSymbol, value)
  def setAnnotationSplitSymbol(value: String): this.type = set(annotationSplitSymbol, value)
  def setCleanAnnotations(value: Boolean): this.type = set(cleanAnnotations, value)
  def setIncludeMetadata(value: Boolean): this.type = set(includeMetadata, value)
  def setOutputAsArray(value: Boolean): this.type = set(outputAsArray, value)

  def getOutputCols: Array[String] = get(outputCols).getOrElse(getInputCols.map("finished_" + _))
  def getInputCols: Array[String] = $(inputCols)
  def getValueSplitSymbol: String = $(valueSplitSymbol)
  def getAnnotationSplitSymbol: String = $(annotationSplitSymbol)
  def getCleanAnnotations: Boolean = $(cleanAnnotations)
  def getIncludeMetadata: Boolean = $(includeMetadata)
  def getOutputAsArray: Boolean = $(outputAsArray)

  setDefault(
    cleanAnnotations -> true,
    includeMetadata -> false,
    outputAsArray -> true)

  def this() = this(Identifiable.randomUID("finisher"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    require(getInputCols.length == getOutputCols.length, "inputCols and outputCols length must match")
    getInputCols.foreach {
      annotationColumn =>
        require(getInputCols.forall(schema.fieldNames.contains),
          s"pipeline annotator stages incomplete. " +
            s"expected: ${getInputCols.mkString(", ")}, " +
            s"found: ${schema.fields.filter(_.dataType == ArrayType(Annotation.dataType)).map(_.name).mkString(", ")}, " +
            s"among available: ${schema.fieldNames.mkString(", ")}")
        require(schema(annotationColumn).dataType == ArrayType(Annotation.dataType),
          s"column [$annotationColumn] must be an NLP Annotation column")
    }
    val metadataFields =  getOutputCols.flatMap(outputCol => {
      if ($(outputAsArray))
        Some(StructField(outputCol + "_md", MapType(StringType, StringType), nullable = false))
      else
        None
    })
    val outputFields = schema.fields ++
      getOutputCols.map(outputCol => {
        if ($(outputAsArray))
          StructField(outputCol, ArrayType(StringType), nullable = false)
        else
          StructField(outputCol, StringType, nullable = true)
      }) ++ metadataFields
    val cleanFields = if ($(cleanAnnotations)) outputFields.filterNot(f =>
      f.dataType == ArrayType(Annotation.dataType)
    ) else outputFields
    StructType(cleanFields)
  }

  override def transform(dataset: Dataset[_]): Dataset[Row] = {
    /*For some reason, Dataset[_] -> Dataset[Row] is not accepted through foldRight
    val flattened = getInputCols.foldRight(dataset)((inputCol, data) =>
      data.withColumn(inputCol, Annotation.flatten(data.col(inputCol))).toDF()
    )
    */
    require(getInputCols.length == getOutputCols.length, "inputCols and outputCols length must match")
    val cols = getInputCols.zip(getOutputCols)
    var flattened = dataset
    cols.foreach { case (inputCol, outputCol) =>
      flattened = {
        flattened.withColumn(
          outputCol, {
            if ($(outputAsArray))
              Annotation.flattenArray(flattened.col(inputCol))
            else if (!$(includeMetadata))
              Annotation.flatten($(valueSplitSymbol), $(annotationSplitSymbol))(flattened.col(inputCol))
            else
              Annotation.flattenDetail($(valueSplitSymbol), $(annotationSplitSymbol))(flattened.col(inputCol))
          }
        )
      }
    }
    if ($(outputAsArray) && $(includeMetadata))
      cols.foreach { case (inputCol, outputCol) =>
        flattened = flattened.withColumn(
          outputCol + "_md",
          Annotation.flattenArrayMetadata(flattened.col(inputCol))
        )
      }
    if ($(cleanAnnotations)) flattened.drop(
      flattened.schema.fields
        .filter(_.dataType == ArrayType(Annotation.dataType))
        .map(_.name):_*)
    else flattened.toDF()
  }

}
object Finisher extends DefaultParamsReadable[Finisher]