package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorType
import org.apache.spark.ml.param.{BooleanParam, IntParam, Params}
import org.apache.spark.sql.Column
import org.apache.spark.sql.types.MetadataBuilder

trait HasEmbeddingsProperties extends Params {

  val dimension = new IntParam(this, "dimension", "Number of embedding dimensions")

  def setDimension(value: Int): this.type = set(this.dimension, value)
  def getDimension: Int = $(dimension)

  protected def wrapEmbeddingsMetadata(col: Column, embeddingsDim: Int, embeddingsRef: Option[String] = None): Column = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", AnnotatorType.WORD_EMBEDDINGS)
    metadataBuilder.putLong("dimension", embeddingsDim.toLong)
    embeddingsRef.foreach(ref => metadataBuilder.putString("ref", ref))
    col.as(col.toString, metadataBuilder.build)
  }

}
