package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

class Chunk2Doc(override val uid: String) extends AnnotatorModel[Chunk2Doc] {

  def this() = this(Identifiable.randomUID("REGEX_TOKENIZER"))

  override val annotatorType: AnnotatorType = DOCUMENT

  override val requiredAnnotatorTypes: Array[String] = Array(CHUNK)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map(annotation => {
      Annotation(
        annotatorType,
        annotation.begin,
        annotation.end,
        annotation.result,
        annotation.metadata
      )
    })
  }

}

object Chunk2Doc extends DefaultParamsReadable[Chunk2Doc]