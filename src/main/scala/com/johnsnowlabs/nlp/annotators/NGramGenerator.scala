package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable, HasSimpleAnnotate}
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, ParamValidators}
import org.apache.spark.ml.util.Identifiable

/**
  * A feature transformer that converts the input array of strings (annotatorType TOKEN) into an
  * array of n-grams (annotatorType CHUNK).
  * Null values in the input array are ignored.
  * It returns an array of n-grams where each n-gram is represented by a space-separated string of
  * words.
  *
  * When the input is empty, an empty array is returned.
  * When the input array length is less than n (number of elements per n-gram), no n-grams are
  * returned.
  *
  * For more extended examples see the [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/chunking/NgramGenerator.ipynb Spark NLP Workshop]]
  * and the [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/NGramGeneratorTestSpec.scala NGramGeneratorTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.SentenceDetector
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.NGramGenerator
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentence = new SentenceDetector()
  *   .setInputCols("document")
  *   .setOutputCol("sentence")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols(Array("sentence"))
  *   .setOutputCol("token")
  *
  * val nGrams = new NGramGenerator()
  *   .setInputCols("token")
  *   .setOutputCol("ngrams")
  *   .setN(2)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *     documentAssembler,
  *     sentence,
  *     tokenizer,
  *     nGrams
  *   ))
  *
  * val data = Seq("This is my sentence.").toDF("text")
  * val results = pipeline.fit(data).transform(data)
  *
  * results.selectExpr("explode(ngrams) as result").show(false)
  * +------------------------------------------------------------+
  * |result                                                      |
  * +------------------------------------------------------------+
  * |[chunk, 0, 6, This is, [sentence -> 0, chunk -> 0], []]     |
  * |[chunk, 5, 9, is my, [sentence -> 0, chunk -> 1], []]       |
  * |[chunk, 8, 18, my sentence, [sentence -> 0, chunk -> 2], []]|
  * |[chunk, 11, 19, sentence ., [sentence -> 0, chunk -> 3], []]|
  * +------------------------------------------------------------+
  * }}}
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
class NGramGenerator (override val uid: String) extends AnnotatorModel[NGramGenerator] with HasSimpleAnnotate[NGramGenerator] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Output annotator type : CHUNK
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = CHUNK
  /** Input annotator type : TOKEN
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("NGRAM_GENERATOR"))

  /**
    * Minimum n-gram length, greater than or equal to 1 (Default: `2`, bigram features)
    *
    * @group param
    */
  val n: IntParam = new IntParam(this, "n", "Number elements per n-gram (>=1)",
    ParamValidators.gtEq(1))

  /** Whether to calculate just the actual n-grams or all n-grams from 1 through n (Default: `false`)
    *
    * @group param
    **/
  val enableCumulative: BooleanParam = new BooleanParam(this, "enableCumulative", "Whether to calculate just the actual n-grams or all n-grams from 1 through n")
  /** Glue character used to join the tokens (Default: `" "`)
    *
    * @group param
    **/
  val delimiter: Param[String] = new Param[String](this, "delimiter", "Glue character used to join the tokens")

  /** Number elements per n-gram (>=1) (Default: `2`)
    *
    * @group setParam
    **/
  def setN(value: Int): this.type = set(n, value)

  /** Whether to calculate just the actual n-grams or all n-grams from 1 through n (Default: `false`)
    *
    * @group setParam
    **/
  def setEnableCumulative(value: Boolean): this.type = set(enableCumulative, value)

  /** Glue character used to join the tokens (Default: `" "`)
    *
    * @group setParam
    **/
  def setDelimiter(value: String): this.type = {
    require(value.length == 1, "Delimiter should have length == 1")
    set(delimiter, value)
  }

  /** Number elements per n-gram (>=1) (Default: `2`)
    *
    * @group getParam
    **/
  def getN: Int = $(n)

  /** Whether to calculate just the actual n-grams or all n-grams from 1 through n (Default: `false`)
    *
    * @group getParam
    **/
  def getEnableCumulative: Boolean = $(enableCumulative)

  /** Glue character used to join the tokens (Default: `" "`)
    *
    * @group getParam
    **/
  def getDelimiter: String = $(delimiter)

  setDefault(
    n -> 2,
    enableCumulative -> false,
    delimiter -> " "
  )

  private def generateNGrams(documents: Seq[(Int, Seq[Annotation])]): Seq[Annotation] = {

    case class NgramChunkAnnotation(currentChunkIdx:Int, annotations: Seq[Annotation])

    val docAnnotation = documents.flatMap { case (idx: Int, annotation: Seq[Annotation]) =>

      val range = if($(enableCumulative)) 1 to $(n) else $(n) to $(n)

      val ngramsAnnotation = range.foldLeft(NgramChunkAnnotation(0,Seq[Annotation]()))((currentNgChunk, k) => {

        val chunksForCurrentWindow = annotation.iterator.sliding(k).withPartial(false).zipWithIndex.map { case (tokens: Seq[Annotation], localChunkIdx: Int) =>
          Annotation(
            outputAnnotatorType,
            tokens.head.begin,
            tokens.last.end,
            tokens.map(_.result).mkString($(delimiter)),
            Map(
              "sentence" -> tokens.head.metadata.getOrElse("sentence", "0"),
              "chunk" -> tokens.head.metadata.getOrElse("chunk", (currentNgChunk.currentChunkIdx + localChunkIdx).toString)
            )
          )
        }.toArray
        NgramChunkAnnotation(currentNgChunk.currentChunkIdx + chunksForCurrentWindow.length, currentNgChunk.annotations++chunksForCurrentWindow)
      })
      ngramsAnnotation.annotations
    }

    docAnnotation
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val documentsWithTokens = annotations
      .filter(token => token.annotatorType == TOKEN)
      .groupBy(_.metadata.getOrElse("sentence", "0").toInt)
      .toSeq
      .sortBy(_._1)

    generateNGrams(documentsWithTokens)

  }
}

object NGramGenerator extends ParamsAndFeaturesReadable[NGramGenerator]