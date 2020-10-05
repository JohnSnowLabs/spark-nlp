package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.common.NerTagged
import com.johnsnowlabs.nlp.annotators.ner.NerTagsEncoding
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.collection.mutable.ArrayBuffer

class TokenAssembler(override val uid: String) extends AnnotatorModel[TokenAssembler] with HasSimpleAnnotate[TokenAssembler] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)

  val preservePosition: BooleanParam = new BooleanParam(this, "preservePosition", "Whether to preserve the actual position of the tokens or reduce them to one space")

  def setPreservePosition(value: Boolean): this.type = set(preservePosition, value)

  setDefault(
    preservePosition -> false
  )

  def this() = this(Identifiable.randomUID("TOKEN_ASSEMBLER"))

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val result = ArrayBuffer[Annotation]()

    val sentences_init = annotations.filter(_.annotatorType == AnnotatorType.DOCUMENT)


    sentences_init.zipWithIndex.foreach { case (sentence, sentenceIndex) =>

      val tokens = annotations.filter(token =>
        token.annotatorType == AnnotatorType.TOKEN &&
          token.begin >= sentence.begin &&
          token.end <= sentence.end)

      var fullSentence: String = ""
      var lastEnding: Int = 0

      tokens.foreach { case (token) =>
        if (token.begin > lastEnding && token.begin - lastEnding != 1 && lastEnding != 0) {
          if ($(preservePosition)) {
            val tokenBreaks = sentence.result.substring(lastEnding + 1 - sentence.begin, token.begin - sentence.begin)
            val matches = ("[\\r\\t\\f\\v\\n ]+".r).findAllIn(tokenBreaks).mkString
            if (matches.length > 0) {
              fullSentence = fullSentence ++ matches ++ token.result
            } else {
              fullSentence = fullSentence ++ " " ++ token.result
            }
          } else {
            fullSentence = fullSentence ++ " " ++ token.result
          }
        } else {
          fullSentence = fullSentence ++ token.result
        }
        lastEnding = token.end
        fullSentence
      }

      val beginIndex = sentence.begin
      val endIndex = fullSentence.length - 1

      val annotation = Annotation(
        DOCUMENT,
        beginIndex,
        beginIndex + endIndex,
        fullSentence,
        Map("sentence" -> sentenceIndex.toString)
      )

      result.append(annotation)
    }
    result
  }

}

object TokenAssembler extends DefaultParamsReadable[TokenAssembler]