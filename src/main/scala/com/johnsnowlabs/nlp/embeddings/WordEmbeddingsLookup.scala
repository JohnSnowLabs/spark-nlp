package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorType._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset


class WordEmbeddingsLookup(override val uid: String)
  extends ApproachWithWordEmbeddings[WordEmbeddingsLookup, WordEmbeddingsLookupModel] {

  def this() = this(Identifiable.randomUID("EMBEDDINGS_LOOKUP"))

  override val description: String = "Indexes embeddings for fast lookup"
  override val outputAnnotatorType: AnnotatorType = WORD_EMBEDDINGS

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): WordEmbeddingsLookupModel = {
    new WordEmbeddingsLookupModel()
  }
}

object WordEmbeddingsLookup extends DefaultParamsReadable[WordEmbeddingsLookup]

