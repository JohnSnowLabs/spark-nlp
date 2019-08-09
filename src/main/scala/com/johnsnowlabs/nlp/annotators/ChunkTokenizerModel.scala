package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, ParamsAndFeaturesReadable}
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.ChunkSplit
import org.apache.spark.ml.util.Identifiable

class ChunkTokenizerModel(override val uid: String) extends TokenizerModel {

  def this() = this(Identifiable.randomUID("CHUNK_TOKENIZER"))

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](CHUNK)

  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** one to many annotation */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = ChunkSplit.unpack(annotations)
    val tokenized = tag(sentences)

    tokenized.zipWithIndex.flatMap{case (sentence, sentenceIndex) =>
      sentence.indexedTokens.map{token =>
        Annotation(outputAnnotatorType, token.begin, token.end, token.token,
          Map("chunk" -> sentenceIndex.toString, "sentence" -> sentence.sentenceIndex.toString))
      }}

  }

}

object ChunkTokenizerModel extends ParamsAndFeaturesReadable[ChunkTokenizerModel]