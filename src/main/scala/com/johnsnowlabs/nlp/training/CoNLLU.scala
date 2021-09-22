/*
 * Copyright 2017-2021 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.training

import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, DocumentAssembler}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Dataset, SparkSession}

case class CoNLLUDocument(text: String,
                          uPosTagged: Seq[PosTaggedSentence],
                          xPosTagged: Seq[PosTaggedSentence],
                          lemma: Seq[PosTaggedSentence]
                        )

/** Instantiates the class to read a CoNLL-U dataset.
 *
 * The dataset should be in the format of [[https://universaldependencies.org/format.html CoNLL-U]]
 * and needs to be specified with `readDataset`, which will create a dataframe with the data.
 *
 * ==Example==
 * {{{
 * import com.johnsnowlabs.nlp.training.CoNLLU
 *
 * val conlluFile = "src/test/resources/conllu/en.test.conllu"
 * val conllDataSet = CoNLLU(false).readDataset(ResourceHelper.spark, conlluFile)
 * conllDataSet.selectExpr("text", "form.result as form", "upos.result as upos", "xpos.result as xpos", "lemma.result as lemma")
 *   .show(1, false)
 * +---------------------------------------+----------------------------------------------+---------------------------------------------+------------------------------+--------------------------------------------+
 * |text                                   |form                                          |upos                                         |xpos                          |lemma                                       |
 * +---------------------------------------+----------------------------------------------+---------------------------------------------+------------------------------+--------------------------------------------+
 * |What if Google Morphed Into GoogleOS?  |[What, if, Google, Morphed, Into, GoogleOS, ?]|[PRON, SCONJ, PROPN, VERB, ADP, PROPN, PUNCT]|[WP, IN, NNP, VBD, IN, NNP, .]|[what, if, Google, morph, into, GoogleOS, ?]|
 * +---------------------------------------+----------------------------------------------+---------------------------------------------+------------------------------+--------------------------------------------+
 * }}}
 * @param explodeSentences Whether to split each sentence into a separate row
 */
case class CoNLLU(explodeSentences: Boolean = true) {

  private val annotationType = ArrayType(Annotation.dataType)

  def readDatasetFromLines(lines: Array[String], spark: SparkSession): Dataset[_] = {
    val docs = CoNLLHelper.readLines(lines, explodeSentences)
    packDocs(docs, spark)
  }

  def readDataset(spark: SparkSession, path: String, readAs: String = ReadAs.TEXT.toString): Dataset[_] = {
    val er = ExternalResource(path, readAs, Map("format" -> "text"))
    val docs = readDocs(er)
    packDocs(docs, spark)
  }

  def packDocs(docs: Seq[CoNLLUDocument], spark: SparkSession): Dataset[_] = {
    import spark.implicits._

    val rows = docs.map { doc =>
      val text = doc.text
      val docs = packAssembly(text)
      val sentences = packSentence(text, doc.uPosTagged)
      val tokenized = packTokenized(doc.uPosTagged)
      val uPosTagged = packPosTagged(doc.uPosTagged)
      val xPosTagged = packPosTagged(doc.xPosTagged)
      val lemma = packTokenized(doc.lemma)

      (text, docs, sentences, tokenized, uPosTagged, xPosTagged, lemma)
    }.toDF.rdd

    spark.createDataFrame(rows, schema)
  }

  def packAssembly(text: String, isTraining: Boolean = true): Seq[Annotation] = {
    new DocumentAssembler()
      .assemble(text, Map("training" -> isTraining.toString))
  }

  def packSentence(text: String, sentences: Seq[TaggedSentence]): Seq[Annotation] = {
    val indexedSentences = sentences.zipWithIndex.map{case (sentence, index) =>
      val start = sentence.indexedTaggedWords.map(t => t.begin).min
      val end = sentence.indexedTaggedWords.map(t => t.end).max
      val sentenceText = text.substring(start, end + 1)
      new Sentence(sentenceText, start, end, index)}

    SentenceSplit.pack(indexedSentences)
  }

  def packTokenized(sentences: Seq[TaggedSentence]): Seq[Annotation] = {
    val tokenizedSentences = sentences.zipWithIndex.map{case (sentence, index) =>
      val tokens = sentence.indexedTaggedWords.map(t =>
        IndexedToken(t.word, t.begin, t.end)
      )
      TokenizedSentence(tokens, index)
    }

    TokenizedWithSentence.pack(tokenizedSentences)
  }

  def packPosTagged(sentences: Seq[TaggedSentence]): Seq[Annotation] = {
    PosTagged.pack(sentences)
  }

  def readDocs(er: ExternalResource): Seq[CoNLLUDocument] = {
    val lines = ResourceHelper.parseLines(er)
    CoNLLHelper.readLines(lines, explodeSentences)
  }

  def schema: StructType = {
    val text = StructField("text", StringType)
    val doc = getAnnotationType("document", AnnotatorType.DOCUMENT)
    val sentence = getAnnotationType("sentence", AnnotatorType.DOCUMENT)
    val token = getAnnotationType(CoNLLUCols.FORM.toString.toLowerCase, AnnotatorType.TOKEN)
    val uPos = getAnnotationType(CoNLLUCols.UPOS.toString.toLowerCase, AnnotatorType.POS)
    val xPos = getAnnotationType(CoNLLUCols.XPOS.toString.toLowerCase, AnnotatorType.POS)
    val lemma = getAnnotationType(CoNLLUCols.LEMMA.toString.toLowerCase, AnnotatorType.TOKEN)

    StructType(Seq(text, doc, sentence, token, uPos, xPos, lemma))
  }

  def getAnnotationType(column: String, annotatorType: String, addMetadata: Boolean = true): StructField = {
    if (!addMetadata)
      StructField(column, annotationType, nullable = false)
    else {
      val metadataBuilder: MetadataBuilder = new MetadataBuilder()
      metadataBuilder.putString("annotatorType", annotatorType)
      StructField(column, annotationType, nullable = false, metadataBuilder.build)
    }
  }

}
