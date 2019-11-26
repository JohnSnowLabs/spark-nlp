package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import org.apache.spark.ml.param.{Param, Params}

trait HasEmbeddingsRef extends Params {

  val embeddingsRef = new Param[String](this, "embeddingsRef", "unique reference name for identification")

  setDefault(embeddingsRef, this.uid)

  def setEmbeddingsRef(value: String): this.type = {
    if (this.isInstanceOf[WordEmbeddingsModel] && get(embeddingsRef).nonEmpty)
      throw new UnsupportedOperationException(s"Cannot override embeddings ref on a WordEmbeddingsModel. Please re-use current ref: $getEmbeddingsRef")
    set(this.embeddingsRef, value)
  }
  def getEmbeddingsRef: String = $(embeddingsRef)

}
