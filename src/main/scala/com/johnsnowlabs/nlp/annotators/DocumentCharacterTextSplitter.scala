package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.functions.ExplodeAnnotations
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, HasSimpleAnnotate}
import org.apache.spark.ml.param.{BooleanParam, IntParam, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame

import scala.util.matching.Regex

/** Annotator which splits large documents into chunks of roughly given size.
  *
  * DocumentCharacterTextSplitter takes a list of separators. It takes the separators in order and
  * splits subtexts if they are over the chunk length, considering optional overlap of the chunks.
  *
  * For example, given chunk size 20 and overlap 5:
  * {{{
  * He was, I take it, the most perfect reasoning and observing machine that the world has seen.
  *
  * ["He was, I take it,", "it, the most", "most perfect", "reasoning and", "and observing", "machine that the", "the world has seen."]
  * }}}
  *
  * Additionally, you can set
  *
  *   - custom patterns with [[setSplitPatterns]]
  *   - whether patterns should be interpreted as regex with [[setPatternsAreRegex]]
  *   - whether to keep the separators with [[setKeepSeparators]]
  *   - whether to trim whitespaces with [[setTrimWhitespace]]
  *   - whether to explode the splits to individual rows with [[setExplodeSplits]]
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/DocumentCharacterTextSplitterTest.scala DocumentCharacterTextSplitterTest]].
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
  * val textSplitter = new DocumentCharacterTextSplitter()
  *   .setInputCols("document")
  *   .setOutputCol("splits")
  *   .setChunkSize(20000)
  *   .setChunkOverlap(200)
  *   .setExplodeSplits(true)
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, textSplitter))
  * val result = pipeline.fit(textDF).transform(textDF)
  *
  * result
  *   .selectExpr(
  *     "splits.result",
  *     "splits[0].begin",
  *     "splits[0].end",
  *     "splits[0].end - splits[0].begin as length")
  *   .show(8, truncate = 80)
  * +--------------------------------------------------------------------------------+---------------+-------------+------+
  * |                                                                          result|splits[0].begin|splits[0].end|length|
  * +--------------------------------------------------------------------------------+---------------+-------------+------+
  * |[ Project Gutenberg's The Adventures of Sherlock Holmes, by Arthur Conan Doyl...|              0|        19994| 19994|
  * |["And Mademoiselle's address?" he asked.\n\n"Is Briony Lodge, Serpentine Aven...|          19798|        39395| 19597|
  * |["How did that help you?"\n\n"It was all-important. When a woman thinks that ...|          39371|        59242| 19871|
  * |["'But,' said I, 'there would be millions of red-headed men who\nwould apply....|          59166|        77833| 18667|
  * |[My friend was an enthusiastic musician, being himself not only a\nvery capab...|          77835|        97769| 19934|
  * |["And yet I am not convinced of it," I answered. "The cases which\ncome to li...|          97771|       117248| 19477|
  * |["Well, she had a slate-coloured, broad-brimmed straw hat, with a\nfeather of...|         117250|       137242| 19992|
  * |["That sounds a little paradoxical."\n\n"But it is profoundly true. Singulari...|         137244|       157171| 19927|
  * +--------------------------------------------------------------------------------+---------------+-------------+------+
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
class DocumentCharacterTextSplitter(override val uid: String)
    extends AnnotatorModel[DocumentCharacterTextSplitter]
    with HasSimpleAnnotate[DocumentCharacterTextSplitter] {

  def this() = this(Identifiable.randomUID("DocumentCharacterTextSplitter"))

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(AnnotatorType.DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.DOCUMENT

  /** Size of each chunk of text
    *
    * @group param
    */
  val chunkSize: IntParam =
    new IntParam(this, "chunkSize", "Size of each chunk of text")

  /** @group setParam */
  def setChunkSize(value: Int): this.type = {
    require(value > 0, "Chunk size should be larger than 0.")
    set(chunkSize, value)
  }

  /** @group setParam */
  def getChunkSize: Int = $(chunkSize)

  /** Length of the overlap between text chunks (Default: `0`)
    *
    * @group param
    */
  val chunkOverlap: IntParam =
    new IntParam(this, "chunkOverlap", "Length of the overlap between text chunks")

  /** @group setParam */
  def setChunkOverlap(value: Int): this.type = {
    require(value <= getChunkSize, "Chunk overlap can't be larger than chunk size.")
    set(chunkOverlap, value)
  }

  /** @group getParam */
  def getChunkOverlap: Int = $(chunkOverlap)

  /** Patterns to separate the text by in decreasing priority (Default: `Array("\n\n", "\n", " ",
    * "")`)
    *
    * Can be interpreted as regular expressions, if `patternsAreRegex` is set to true.
    *
    * @group param
    */
  val splitPatterns: StringArrayParam =
    new StringArrayParam(
      this,
      "splitPatterns",
      "Patterns to separate the text by in decreasing priority")

  /** @group setParam */
  def setSplitPatterns(value: Array[String]): this.type = {
    require(value.nonEmpty, "Patterns are empty")
    set(splitPatterns, value)
  }

  /** @group getParam */
  def getSplitPatterns: Array[String] = $(splitPatterns)

  /** Whether to interpret the split patterns as regular expressions (Default: `false`)
    *
    * @group param
    */
  val patternsAreRegex: BooleanParam =
    new BooleanParam(
      this,
      "patternsAreRegex",
      "Whether to interpret the split patterns as regular expressions")

  /** @group setParam */
  def setPatternsAreRegex(value: Boolean): this.type = set(patternsAreRegex, value)

  /** @group getParam */
  def getPatternsAreRegex: Boolean = $(patternsAreRegex)

  /** Whether to keep the separators in the final result (Default: `false`)
    *
    * @group param
    */
  val keepSeparators: BooleanParam =
    new BooleanParam(this, "keepSeparators", "Whether to keep the separators in the final result")

  /** @group setParam */
  def setKeepSeparators(value: Boolean): this.type = set(keepSeparators, value)

  /** @group getParam */
  def getKeepSeparators: Boolean = $(keepSeparators)

  /** Whether to explode split chunks to separate rows (Default: `false`)
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
    * @group param
    */
  val trimWhitespace: BooleanParam =
    new BooleanParam(this, "trimWhitespace", "Whether to trim whitespaces of extracted chunks")

  /** @group setParam */
  def setTrimWhitespace(value: Boolean): this.type = set(trimWhitespace, value)

  /** @group getParam */
  def getTrimWhitespace: Boolean = $(trimWhitespace)

  setDefault(
    chunkOverlap -> 0,
    explodeSplits -> false,
    keepSeparators -> true,
    patternsAreRegex -> false,
    splitPatterns -> Array("\n\n", "\n", " ", ""),
    trimWhitespace -> true)

  /** Takes a document and annotations and produces new annotations of this annotator's annotation
    * type
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
        getChunkSize,
        getChunkOverlap,
        getKeepSeparators,
        getPatternsAreRegex,
        getTrimWhitespace)

    annotations.zipWithIndex
      .flatMap { case (annotation, i) =>
        val text = annotation.result

        val textChunks = textSplitter.splitText(text, getSplitPatterns)

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
              annotation.metadata ++ Map("document" -> index.toString),
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
