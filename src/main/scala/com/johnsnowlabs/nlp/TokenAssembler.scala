package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.collection.mutable.ArrayBuffer

/**
  * This transformer reconstructs a `DOCUMENT` type annotation from tokens, usually after these have been normalized,
  * lemmatized, normalized, spell checked, etc, in order to use this document annotation in further annotators.
  * Requires `DOCUMENT` and `TOKEN` type annotations as input.
  *
  * For more extended examples on document pre-processing see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/databricks_notebooks/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers_v3.0.ipynb Spark NLP Workshop]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.SentenceDetector
  * import com.johnsnowlabs.nlp.annotator.Tokenizer
  * import com.johnsnowlabs.nlp.annotator.{Normalizer, StopWordsCleaner}
  * import com.johnsnowlabs.nlp.TokenAssembler
  * import org.apache.spark.ml.Pipeline
  *
  * // First, the text is tokenized and cleaned
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentenceDetector = new SentenceDetector()
  *   .setInputCols("document")
  *   .setOutputCol("sentences")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("sentences")
  *   .setOutputCol("token")
  *
  * val normalizer = new Normalizer()
  *   .setInputCols("token")
  *   .setOutputCol("normalized")
  *   .setLowercase(false)
  *
  * val stopwordsCleaner = new StopWordsCleaner()
  *   .setInputCols("normalized")
  *   .setOutputCol("cleanTokens")
  *   .setCaseSensitive(false)
  *
  * // Then the TokenAssembler turns the cleaned tokens into a `DOCUMENT` type structure.
  * val tokenAssembler = new TokenAssembler()
  *   .setInputCols("sentences", "cleanTokens")
  *   .setOutputCol("cleanText")
  *
  * val data = Seq("Spark NLP is an open-source text processing library for advanced natural language processing.")
  *   .toDF("text")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   sentenceDetector,
  *   tokenizer,
  *   normalizer,
  *   stopwordsCleaner,
  *   tokenAssembler
  * )).fit(data)
  *
  * val result = pipeline.transform(data)
  * result.select("cleanText").show(false)
  * +---------------------------------------------------------------------------------------------------------------------------+
  * |cleanText                                                                                                                  |
  * +---------------------------------------------------------------------------------------------------------------------------+
  * |[[document, 0, 80, Spark NLP opensource text processing library advanced natural language processing, [sentence -> 0], []]]|
  * +---------------------------------------------------------------------------------------------------------------------------+
  * }}}
  *
  * @see [[DocumentAssembler]] on the data structure
  * @param uid required uid for storing annotator to disk
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
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class TokenAssembler(override val uid: String) extends AnnotatorModel[TokenAssembler] with HasSimpleAnnotate[TokenAssembler] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /**
    * Output annotator types: DOCUMENT
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  /**
    * Input annotator types: DOCUMENT, TOKEN
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)

  /**
    * Whether to preserve the actual position of the tokens or reduce them to one space (Default: `false`)
    * @group param
    */
  val preservePosition: BooleanParam = new BooleanParam(this, "preservePosition", "Whether to preserve the actual position of the tokens or reduce them to one space")

  /**
    * Whether to preserve the actual position of the tokens or reduce them to one space (Default: `false`)
    * @group setParam
    */
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
      var lastEnding: Int = -1

      tokens.foreach { case (token) =>
        if (token.begin > lastEnding && token.begin - lastEnding != 1 && lastEnding != -1) {
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