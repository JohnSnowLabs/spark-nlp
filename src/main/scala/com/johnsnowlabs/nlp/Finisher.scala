package com.johnsnowlabs.nlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}

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
  protected val includeKeys: BooleanParam =
    new BooleanParam(this, "includeKeys", "annotation metadata format")

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)
  def setInputCols(value: String*): this.type = setInputCols(value.toArray)
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)
  def setOutputCols(value: String*): this.type = setOutputCols(value.toArray)
  def setValueSplitSymbol(value: String): this.type = set(valueSplitSymbol, value)
  def setAnnotationSplitSymbol(value: String): this.type = set(annotationSplitSymbol, value)
  def setCleanAnnotations(value: Boolean): this.type = set(cleanAnnotations, value)
  def setMetadataFormat(value: Boolean): this.type = set(includeKeys, value)

  def getOutputCols: Array[String] = get(outputCols).getOrElse(getInputCols.map("finished_" + _))
  def getInputCols: Array[String] = $(inputCols)
  def getValueSplitSymbol: String = $(valueSplitSymbol)
  def getAnnotationSplitSymbol: String = $(annotationSplitSymbol)
  def getCleanAnnotations: Boolean = $(cleanAnnotations)
  def getIncludeKeys: Boolean = $(includeKeys)

  setDefault(valueSplitSymbol, "#")
  setDefault(annotationSplitSymbol, "@")
  setDefault(cleanAnnotations, true)
  setDefault(includeKeys, false)

  def this() = this(Identifiable.randomUID("document"))

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
    val outputFields = schema.fields ++
      getOutputCols.map(outputCol => StructField(outputCol, StringType, nullable = false))
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
            if (!$(includeKeys))
              Annotation.flatten($(valueSplitSymbol), $(annotationSplitSymbol))(flattened.col(inputCol))
            else
              Annotation.flattenKV($(valueSplitSymbol), $(annotationSplitSymbol))(flattened.col(inputCol))
          }
        )
      }
    }
    if ($(cleanAnnotations)) flattened.drop(
      flattened.schema.fields
        .filter(_.dataType == ArrayType(Annotation.dataType))
        .map(_.name):_*)
    else flattened.toDF()
  }

}
object Finisher extends DefaultParamsReadable[Finisher]