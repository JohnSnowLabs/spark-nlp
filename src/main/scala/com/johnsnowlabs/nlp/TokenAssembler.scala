package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

class TokenAssembler(override val uid: String) extends AnnotatorModel[TokenAssembler]{

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  val preservePosition: BooleanParam = new BooleanParam(this, "preservePosition", "Whether to preserve the actual position of the tokens or reduce them to one space")
  def setPreservePosition(value: Boolean): this.type = set(preservePosition, value)

  setDefault(
    preservePosition -> false
  )

  def this() = this(Identifiable.randomUID("TOKEN_ASSEMBLER"))

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val sentences = annotations
      .groupBy(_.metadata.getOrElse("sentence", "0").toInt)
      .toSeq
      .sortBy(_._1)

    sentences.map{case (idx, sentence) =>
      var fullSentence: String = ""
      var lastEnding: Int = sentence.head.end

      sentence.map{
        token =>
          if(token.begin > lastEnding && token.begin - lastEnding != 1){
            if($(preservePosition)){
              val spaces = Array.fill((token.begin - lastEnding) - 1)(" ").mkString(" ")
              fullSentence = fullSentence ++ spaces ++ token.result
            }else {
              fullSentence = fullSentence ++ " " ++ token.result
            }
          } else{
            fullSentence = fullSentence ++ token.result
          }
          lastEnding = token.end
          fullSentence
      }
      val beginIndex = sentence.head.begin
      val endIndex = fullSentence.length - 1

      Annotation(
        DOCUMENT,
        beginIndex,
        beginIndex+endIndex,
        fullSentence,
        Map("sentence"-> idx.toString)
      )
    }
  }

}

object TokenAssembler extends DefaultParamsReadable[TokenAssembler]