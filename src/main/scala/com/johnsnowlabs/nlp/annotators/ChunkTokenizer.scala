package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, TOKEN}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/** Tokenizes and flattens extracted NER chunks.
 *
 * The ChunkTokenizer will split the extracted NER `CHUNK` type Annotations and will create `TOKEN` type Annotations.
 * The result is then flattened, resulting in a single array.
 *
 * For extended examples of usage, see the [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ChunkTokenizerTestSpec.scala ChunkTokenizerTestSpec]].
 *
 * ==Example==
 * {{{
 * import spark.implicits._
 * import com.johnsnowlabs.nlp.DocumentAssembler
 * import com.johnsnowlabs.nlp.annotators.{ChunkTokenizer, TextMatcher, Tokenizer}
 * import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
 * import com.johnsnowlabs.nlp.util.io.ReadAs
 * import org.apache.spark.ml.Pipeline
 *
 * val documentAssembler = new DocumentAssembler()
 *   .setInputCol("text")
 *   .setOutputCol("document")
 *
 * val sentenceDetector = new SentenceDetector()
 *   .setInputCols(Array("document"))
 *   .setOutputCol("sentence")
 *
 * val tokenizer = new Tokenizer()
 *   .setInputCols(Array("sentence"))
 *   .setOutputCol("token")
 *
 * val entityExtractor = new TextMatcher()
 *   .setInputCols("sentence", "token")
 *   .setEntities("src/test/resources/entity-extractor/test-chunks.txt", ReadAs.TEXT)
 *   .setOutputCol("entity")
 *
 * val chunkTokenizer = new ChunkTokenizer()
 *   .setInputCols("entity")
 *   .setOutputCol("chunk_token")
 *
 * val pipeline = new Pipeline().setStages(Array(
 *     documentAssembler,
 *     sentenceDetector,
 *     tokenizer,
 *     entityExtractor,
 *     chunkTokenizer
 *   ))
 *
 * val data = Seq(
 *   "Hello world, my name is Michael, I am an artist and I work at Benezar",
 *   "Robert, an engineer from Farendell, graduated last year. The other one, Lucas, graduated last week."
 * ).toDF("text")
 * val result = pipeline.fit(data).transform(data)
 *
 * result.selectExpr("entity.result as entity" , "chunk_token.result as chunk_token").show(false)
 * +-----------------------------------------------+---------------------------------------------------+
 * |entity                                         |chunk_token                                        |
 * +-----------------------------------------------+---------------------------------------------------+
 * |[world, Michael, work at Benezar]              |[world, Michael, work, at, Benezar]                |
 * |[engineer from Farendell, last year, last week]|[engineer, from, Farendell, last, year, last, week]|
 * +-----------------------------------------------+---------------------------------------------------+
 * }}}
 *
 * @param uid required internal uid for saving annotator
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
 *
 * */
class ChunkTokenizer(override val uid: String) extends Tokenizer {

  def this() = this(Identifiable.randomUID("CHUNK_TOKENIZER"))

  /** Input Annotator Type : CHUNK
   *
   * @group anno
   * */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](CHUNK)
  /** Output Annotator Type : TOKEN
   *
   * @group anno
   * */
  override val outputAnnotatorType: AnnotatorType = TOKEN

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): TokenizerModel = {
    val ruleFactory = buildRuleFactory

    val processedExceptions = get(exceptionsPath)
      .map(er => ResourceHelper.parseLines(er))
      .getOrElse(Array.empty[String]) ++ get(exceptions).getOrElse(Array.empty[String])

    val raw = new ChunkTokenizerModel()
      .setCaseSensitiveExceptions($(caseSensitiveExceptions))
      .setTargetPattern($(targetPattern))
      .setRules(ruleFactory)

    if (processedExceptions.nonEmpty)
      raw.setExceptions(processedExceptions)
    else
      raw
  }

}

/**
 * This is the companion object of [[ChunkTokenizer]]. Please refer to that class for the documentation.
 */
object ChunkTokenizer extends DefaultParamsReadable[ChunkTokenizer]
