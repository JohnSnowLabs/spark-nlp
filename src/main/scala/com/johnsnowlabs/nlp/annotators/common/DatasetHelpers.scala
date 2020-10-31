package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.ml.tensorflow.SentenceGrouper
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.catalyst.encoders.RowEncoder

import scala.reflect.ClassTag

object DatasetHelpers {

  implicit class DataFrameHelper(dataset: DataFrame) {
    def randomize: DataFrame = {
      implicit val encoder = RowEncoder(dataset.schema)
      dataset.mapPartitions {
        new scala.util.Random().shuffle(_).toIterator
      }
    }
  }
  
  def doSlice[T: ClassTag](dataset: TraversableOnce[T], getLen: T => Int, batchSize: Int = 32): Iterator[Array[T]] = {
    val gr = SentenceGrouper[T](getLen)
    gr.slice(dataset, batchSize)
  }

  def slice(dataset: TraversableOnce[(TextSentenceLabels, WordpieceEmbeddingsSentence)], batchSize: Int = 32):
  Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] = {
    doSlice[(TextSentenceLabels, WordpieceEmbeddingsSentence)](dataset, _._2.tokens.length, batchSize)
  }

}
