package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

class Chunk2Doc(override val uid: String) extends AnnotatorModel[Chunk2Doc] with WithAnnotate[Chunk2Doc] {

  def this() = this(Identifiable.randomUID("CHUNK2DOC"))

  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  override val inputAnnotatorTypes: Array[String] = Array(CHUNK)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map(annotation => {
      Annotation(
        outputAnnotatorType,
        annotation.begin,
        annotation.end,
        annotation.result,
        annotation.metadata
      )
    })
  }

}

object Chunk2Doc extends DefaultParamsReadable[Chunk2Doc]