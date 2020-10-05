package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, WithAnnotate}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  *
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  *
  **/
class Token2Chunk(override val uid: String) extends AnnotatorModel[Token2Chunk] with WithAnnotate[Token2Chunk] {


  /** Output Annotator Type : CHUNK
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = CHUNK
  /** Input Annotator Type : TOKEN
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("TOKEN2CHUNK"))

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map { token =>
      Annotation(
        CHUNK,
        token.begin,
        token.end,
        token.result,
        token.metadata
      )
    }
  }

}

object Token2Chunk extends DefaultParamsReadable[Token2Chunk]