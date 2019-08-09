package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, TOKEN}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class ChunkTokenizer(override val uid: String) extends Tokenizer {

  def this() = this(Identifiable.randomUID("CHUNK_TOKENIZER"))

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](CHUNK)

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

object ChunkTokenizer extends DefaultParamsReadable[ChunkTokenizer]
