package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.Dataset

trait HasEmbeddingsRef extends Params {

  val embeddingsRef = new Param[String](this, "embeddingsRef", "unique reference name for identification")

  def setEmbeddingsRef(value: String): this.type = {
    if (this.isInstanceOf[WordEmbeddingsModel] && get(embeddingsRef).nonEmpty)
      throw new UnsupportedOperationException(s"Cannot override embeddings ref on a WordEmbeddingsModel. " +
        s"Please re-use current ref: $getEmbeddingsRef")
    set(this.embeddingsRef, value)
  }
  def getEmbeddingsRef: String = $(embeddingsRef)

  def validateEmbeddingsRef(dataset: Dataset[_], inputCols: Array[String]): Unit = {
    val embeddings_col = dataset.schema.fields
      .find(f => inputCols.contains(f.name) && f.metadata.getString("annotatorType") == AnnotatorType.WORD_EMBEDDINGS)
      .getOrElse(throw new Exception("Could not find a valid embeddings column")).name

    val embeddings_meta = dataset.select(embeddings_col).schema.fields.head.metadata

    require(embeddings_meta.contains("ref"), "Cannot find embeddings ref in column schema metadata")

    require(embeddings_meta.getString("ref") == $(embeddingsRef),
      s"Found embeddings column, but embeddings ref does not match to the embeddings this model was trained with. " +
        s"Make sure you are using the right embeddings in your pipeline, with ref: ${$(embeddingsRef)}")
  }

}
