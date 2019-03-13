package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorType
import org.apache.spark.ml.param.{BooleanParam, IntParam, Params}
import org.apache.spark.sql.Column
import org.apache.spark.sql.types.MetadataBuilder

trait HasEmbeddings extends Params {

  val dimension = new IntParam(this, "dimension", "Number of embedding dimensions")
  val caseSensitive = new BooleanParam(this, "caseSensitive", "whether to ignore case in tokens for embeddings matching")

  setDefault(caseSensitive, false)

  def setDimension(value: Int): this.type = set(this.dimension, value)
  def setCaseSensitive(value: Boolean): this.type = set(this.caseSensitive, value)

  def getDimension: Int = $(dimension)
  def getCaseSensitive: Boolean = $(caseSensitive)

  protected def wrapEmbeddingsMetadata(col: Column, embeddingsDim: Int): Column = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", AnnotatorType.WORD_EMBEDDINGS)
    metadataBuilder.putLong("dimension", embeddingsDim.toLong)
    col.as(col.toString, metadataBuilder.build)
  }

}
