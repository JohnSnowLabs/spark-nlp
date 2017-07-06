package com.jsl.nlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{StructField, StructType}

/**
  * Created by saif on 06/07/17.
  */

class DocumentAssembler(override val uid: String) extends Transformer with DefaultParamsWritable {

  val outputColName: Param[String] = new Param[String](this, "outputColName", "document column name")

  val inputColName: Param[String] = new Param[String](this, "inputColName", "input text column for processing")

  val idColName: Param[String] = new Param[String](this, "idColName", "id column for row reference")

  val metadataColName: Param[String] = new Param[String](this, "metadataColName", "metadata for document column")

  def setOutputColName(value: String): this.type = set(outputColName, value)

  def getOutputColName: String = get(outputColName).getOrElse("document")

  def setInputColName(value: String): this.type = set(inputColName, value)

  def getInputColName: String = $(inputColName)

  def setIdColName(value: String): this.type = set(idColName, value)

  def getIdColName: String = $(idColName)

  def setMetadataColName(value: String): this.type = set(metadataColName, value)

  def getMetadataColName: String = $(metadataColName)

  def this() = this(Identifiable.randomUID("doc"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.withColumn(
      getOutputColName,
      Document.column(dataset(getInputColName))(get(idColName).map(dataset.col), get(metadataColName).map(dataset.col))
    )
  }

  override def transformSchema(schema: StructType): StructType = {
    require(schema.fieldNames.contains(getInputColName), s"$getInputColName not within ${schema.fieldNames.mkString(", ")}")
    val outputFields = schema.fields :+
      StructField(getOutputColName, Document.dataType, nullable = false)
    StructType(outputFields)
  }

}
object DocumentAssembler extends DefaultParamsReadable[DocumentAssembler]