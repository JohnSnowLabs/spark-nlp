package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.CHUNK
import com.johnsnowlabs.nlp.annotators.common.{ChunkSplit, TokenizedWithSentence}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

class ChunkTokenizer(override val uid: String) extends Tokenizer {

  def this() = this(Identifiable.randomUID("CHUNK_TOKENIZER"))

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](CHUNK)

  /** one to many annotation */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = ChunkSplit.unpack(annotations)
    val tokenized = tag(sentences)
    TokenizedWithSentence.pack(tokenized)
  }

}

object ChunkTokenizer extends DefaultParamsReadable[ChunkTokenizer]