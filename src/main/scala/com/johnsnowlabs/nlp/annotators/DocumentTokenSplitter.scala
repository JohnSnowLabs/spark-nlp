package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, HasSimpleAnnotate}
import org.apache.spark.ml.param.{BooleanParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame

import scala.util.matching.Regex
import com.johnsnowlabs.nlp.functions.ExplodeAnnotations

/** Annotator that splits large documents into smaller documents based on the number of tokens in
  * the text.
  *
  * Currently, DocumentTokenSplitter splits the text by whitespaces to create the tokens. The
  * number of these tokens will then be used as a measure of the text length. In the future, other
  * tokenization techniques will be supported.
  *
  * For example, given 3 tokens and overlap 1:
  * {{{
  * He was, I take it, the most perfect reasoning and observing machine that the world has seen.
  *
  * ["He was, I", "I take it,", "it, the most", "most perfect reasoning", "reasoning and observing", "observing machine that", "that the world", "world has seen."]
  * }}}
  *
  * Additionally, you can set
  *
  *   - whether to trim whitespaces with [[setTrimWhitespace]]
  *   - whether to explode the splits to individual rows with [[setExplodeSplits]]
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/DocumentTokenSplitterTest.scala DocumentTokenSplitterTest]].
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.annotator._
  * import com.johnsnowlabs.nlp.DocumentAssembler
  * import org.apache.spark.ml.Pipeline
  *
  * val textDF =
  *   spark.read
  *     .option("wholetext", "true")
  *     .text("src/test/resources/spell/sherlockholmes.txt")
  *     .toDF("text")
  *
  * val documentAssembler = new DocumentAssembler().setInputCol("text")
  * val textSplitter = new DocumentTokenSplitter()
  *   .setInputCols("document")
  *   .setOutputCol("splits")
  *   .setNumTokens(512)
  *   .setTokenOverlap(10)
  *   .setExplodeSplits(true)
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, textSplitter))
  * val result = pipeline.fit(textDF).transform(textDF)
  *
  * result
  *   .selectExpr(
  *     "splits.result as result",
  *     "splits[0].begin as begin",
  *     "splits[0].end as end",
  *     "splits[0].end - splits[0].begin as length",
  *     "splits[0].metadata.numTokens as tokens")
  *   .show(8, truncate = 80)
  * +--------------------------------------------------------------------------------+-----+-----+------+------+
  * |                                                                          result|begin|  end|length|tokens|
  * +--------------------------------------------------------------------------------+-----+-----+------+------+
  * |[ Project Gutenberg's The Adventures of Sherlock Holmes, by Arthur Conan Doyl...|    0| 3018|  3018|   512|
  * |[study of crime, and occupied his\nimmense faculties and extraordinary powers...| 2950| 5707|  2757|   512|
  * |[but as I have changed my clothes I can't imagine how you\ndeduce it. As to M...| 5659| 8483|  2824|   512|
  * |[quarters received. Be in your chamber then at that hour, and do\nnot take it...| 8427|11241|  2814|   512|
  * |[a pity\nto miss it."\n\n"But your client--"\n\n"Never mind him. I may want y...|11188|13970|  2782|   512|
  * |[person who employs me wishes his agent to be unknown to\nyou, and I may conf...|13918|16898|  2980|   512|
  * |[letters back."\n\n"Precisely so. But how--"\n\n"Was there a secret marriage?...|16836|19744|  2908|   512|
  * |[seven hundred in\nnotes," he said.\n\nHolmes scribbled a receipt upon a shee...|19683|22551|  2868|   512|
  * +--------------------------------------------------------------------------------+-----+-----+------+------+
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
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
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class DocumentTokenSplitter(override val uid: String)
    extends AnnotatorModel[DocumentTokenSplitter]
    with HasSimpleAnnotate[DocumentTokenSplitter] {

  def this() = this(Identifiable.randomUID("DocumentTokenSplitter"))

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(AnnotatorType.DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.DOCUMENT

  /** Limit of the number of tokens in a text
    *
    * @group param
    */
  val numTokens: IntParam =
    new IntParam(this, "numTokens", "Limit of the number of tokens in a text")

  /** @group setParam */
  def setNumTokens(value: Int): this.type = {
    require(value > 0, "Number of tokens should be larger than 0.")
    set(numTokens, value)
  }

  /** @group setParam */
  def getNumTokens: Int = $(numTokens)

  /** Length of the token overlap between text chunks (Default: `0`)
    *
    * @group param
    */
  val tokenOverlap: IntParam =
    new IntParam(this, "tokenOverlap", "Length of the overlap between text chunks")

  /** @group setParam */
  def setTokenOverlap(value: Int): this.type = {
    require(value <= getNumTokens, "Token overlap can't be larger than number of tokens.")
    set(tokenOverlap, value)
  }

  /** @group getParam */
  def getTokenOverlap: Int = $(tokenOverlap)

  /** Whether to explode split chunks to separate rows
    *
    * @group param
    */
  val explodeSplits: BooleanParam =
    new BooleanParam(this, "explodeSplits", "Whether to explode split chunks to separate rows")

  /** @group setParam */
  def setExplodeSplits(value: Boolean): this.type = set(explodeSplits, value)

  /** @group getParam */
  def getExplodeSplits: Boolean = $(explodeSplits)

  /** Whether to trim whitespaces of extracted chunks (Default: `true`)
    *
    * @group param
    */
  val trimWhitespace: BooleanParam =
    new BooleanParam(this, "trimWhitespace", "Whether to trim whitespaces of extracted chunks")

  /** @group setParam */
  def setTrimWhitespace(value: Boolean): this.type = set(trimWhitespace, value)

  /** @group getParam */
  def getTrimWhitespace: Boolean = $(trimWhitespace)

  setDefault(tokenOverlap -> 0, explodeSplits -> false, trimWhitespace -> true)

  // Replaced by the desired tokenizer in the future
  private val tokenSplitPattern = "\\s+".r

  def lengthFromTokens(text: String): Int =
    tokenSplitPattern.split(text).count(_.nonEmpty)

  /** Takes a Document and produces document splits based on a Tokenizers
    *
    * @param annotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val textSplitter =
      new TextSplitter(
        chunkSize = getNumTokens,
        chunkOverlap = getTokenOverlap,
        keepSeparators = true,
        patternsAreRegex = true,
        trimWhitespace = getTrimWhitespace,
        lengthFunction = lengthFromTokens)

    val documentSplitPatterns = Array("\\s+")

    annotations.zipWithIndex
      .flatMap { case (annotation, i) =>
        val text = annotation.result

        val textChunks = textSplitter.splitText(text, documentSplitPatterns)

        textChunks.zipWithIndex.map { case (textChunk, index) =>
          val textChunkBegin = Regex.quote(textChunk).r.findFirstMatchIn(text) match {
            case Some(m) => m.start
            case None => -1
          }
          val textChunkEnd = if (textChunkBegin >= 0) textChunkBegin + textChunk.length else -1

          (
            i,
            new Annotation(
              AnnotatorType.DOCUMENT,
              textChunkBegin,
              textChunkEnd,
              textChunk,
              annotation.metadata ++ Map(
                "document" -> index.toString,
                "numTokens" -> lengthFromTokens(textChunk).toString),
              annotation.embeddings))
        }
      }
      .sortBy(_._1)
      .map(_._2)
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    if (getExplodeSplits) dataset.explodeAnnotationsCol(getOutputCol, getOutputCol) else dataset
  }
}
