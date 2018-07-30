package com.johnsnowlabs.nlp

import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by saif on 06/07/17.
  */

class ChunkAssembler(override val uid: String) extends AnnotatorModel[ChunkAssembler]{

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val annotatorType: AnnotatorType = DOCUMENT

  override val requiredAnnotatorTypes: Array[String] = Array(CHUNK)

  def this() = this(Identifiable.randomUID("CHUNK_ASSEMBLER"))

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map(annotation => {
      Annotation(
        DOCUMENT,
        annotation.begin,
        annotation.end,
        annotation.result,
        annotation.metadata
      )
    })
  }

}
object ChunkAssembler extends DefaultParamsReadable[ChunkAssembler]
