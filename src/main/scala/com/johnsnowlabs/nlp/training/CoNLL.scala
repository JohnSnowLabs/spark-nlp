/*
 * Copyright 2017-2022 John Snow Labs
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

import com.johnsnowlabs.nlp.annotators.common.Annotated.{NerTaggedSentence, PosTaggedSentence}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, DocumentAssembler}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer

case class CoNLLDocument(
    text: String,
    nerTagged: Seq[NerTaggedSentence],
    posTagged: Seq[PosTaggedSentence],
    docId: Option[String])

/** Helper class to load a CoNLL type dataset for training.
  *
  * The dataset should be in the format of
  * [[https://www.clips.uantwerpen.be/conll2003/ner/ CoNLL 2003]] and needs to be specified with
  * `readDataset`. Other CoNLL datasets are not supported.
  *
  * Two types of input paths are supported,
  *
  * Folder: this is a path ending in `*`, and representing a collection of CoNLL files within a
  * directory. E.g., 'path/to/multiple/conlls&#47;*' Using this pattern will result in all the
  * files being read into a single Dataframe. Some constraints apply on the schemas of the
  * multiple files.
  *
  * File: this is a path to a single file. E.g., 'path/to/single_file.conll'
  *
  * ==Example==
  * {{{
  * val trainingData = CoNLL().readDataset(spark, "src/test/resources/conll2003/eng.train")
  * trainingData.selectExpr("text", "token.result as tokens", "pos.result as pos", "label.result as label")
  *   .show(3, false)
  * +------------------------------------------------+----------------------------------------------------------+-------------------------------------+-----------------------------------------+
  * |text                                            |tokens                                                    |pos                                  |label                                    |
  * +------------------------------------------------+----------------------------------------------------------+-------------------------------------+-----------------------------------------+
  * |EU rejects German call to boycott British lamb .|[EU, rejects, German, call, to, boycott, British, lamb, .]|[NNP, VBZ, JJ, NN, TO, VB, JJ, NN, .]|[B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]|
  * |Peter Blackburn                                 |[Peter, Blackburn]                                        |[NNP, NNP]                           |[B-PER, I-PER]                           |
  * |BRUSSELS 1996-08-22                             |[BRUSSELS, 1996-08-22]                                    |[NNP, CD]                            |[B-LOC, O]                               |
  * +------------------------------------------------+----------------------------------------------------------+-------------------------------------+-----------------------------------------+
  *
  * trainingData.printSchema
  * root
  *  |-- text: string (nullable = true)
  *  |-- document: array (nullable = false)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- annotatorType: string (nullable = true)
  *  |    |    |-- begin: integer (nullable = false)
  *  |    |    |-- end: integer (nullable = false)
  *  |    |    |-- result: string (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
  *  |    |    |-- embeddings: array (nullable = true)
  *  |    |    |    |-- element: float (containsNull = false)
  *  |-- sentence: array (nullable = false)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- annotatorType: string (nullable = true)
  *  |    |    |-- begin: integer (nullable = false)
  *  |    |    |-- end: integer (nullable = false)
  *  |    |    |-- result: string (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
  *  |    |    |-- embeddings: array (nullable = true)
  *  |    |    |    |-- element: float (containsNull = false)
  *  |-- token: array (nullable = false)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- annotatorType: string (nullable = true)
  *  |    |    |-- begin: integer (nullable = false)
  *  |    |    |-- end: integer (nullable = false)
  *  |    |    |-- result: string (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
  *  |    |    |-- embeddings: array (nullable = true)
  *  |    |    |    |-- element: float (containsNull = false)
  *  |-- pos: array (nullable = false)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- annotatorType: string (nullable = true)
  *  |    |    |-- begin: integer (nullable = false)
  *  |    |    |-- end: integer (nullable = false)
  *  |    |    |-- result: string (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
  *  |    |    |-- embeddings: array (nullable = true)
  *  |    |    |    |-- element: float (containsNull = false)
  *  |-- label: array (nullable = false)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- annotatorType: string (nullable = true)
  *  |    |    |-- begin: integer (nullable = false)
  *  |    |    |-- end: integer (nullable = false)
  *  |    |    |-- result: string (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
  *  |    |    |-- embeddings: array (nullable = true)
  *  |    |    |    |-- element: float (containsNull = false)
  * }}}
  *
  * @param documentCol
  *   Name of the `DOCUMENT` Annotator type column
  * @param sentenceCol
  *   Name of the Sentences of `DOCUMENT` Annotator type column
  * @param tokenCol
  *   Name of the `TOKEN` Annotator type column
  * @param posCol
  *   Name of the `POS` Annotator type column
  * @param conllLabelIndex
  *   Index of the column for NER Label in the dataset
  * @param conllPosIndex
  *   Index of the column for the POS tags in the dataset
  * @param conllDocIdCol
  *   Name of the column for the text in the dataset
  * @param conllTextCol
  *   Name of the column for the text in the dataset
  * @param labelCol
  *   Name of the `NAMED_ENTITY` Annotator type column
  * @param explodeSentences
  *   Whether to explode each sentence to a separate row
  * @param delimiter
  *   Delimiter used to separate columns inside CoNLL file
  * @param includeDocId
  *   Whether to try and parse the document id from the third item in the -DOCSTART- line (X if
  *   not found)
  */
case class CoNLL(
    documentCol: String = "document",
    sentenceCol: String = "sentence",
    tokenCol: String = "token",
    posCol: String = "pos",
    conllLabelIndex: Int = 3,
    conllPosIndex: Int = 1,
    conllDocIdCol: String = "doc_id",
    conllTextCol: String = "text",
    labelCol: String = "label",
    explodeSentences: Boolean = true,
    delimiter: String = " ",
    includeDocId: Boolean = false) {
  /*
    Reads Dataset in CoNLL format and pack it into docs
   */
  def readDocs(er: ExternalResource): Seq[CoNLLDocument] = {
    val lines = ResourceHelper.parseLines(er)

    readLines(lines)
  }

  def clearTokens(tokens: Array[IndexedTaggedWord]): Array[IndexedTaggedWord] = {
    tokens.filter(t => t.word.trim().nonEmpty)
  }

  def readLines(lines: Array[String]): Seq[CoNLLDocument] = {
    var docId: Option[String] = None
    val doc = new StringBuilder()
    val lastSentence = ArrayBuffer.empty[(IndexedTaggedWord, IndexedTaggedWord)]

    val sentences = ArrayBuffer.empty[(TaggedSentence, TaggedSentence)]

    def addSentence(): Unit = {
      val nerTokens = clearTokens(lastSentence.map(t => t._1).toArray)
      val posTokens = clearTokens(lastSentence.map(t => t._2).toArray)

      if (nerTokens.nonEmpty) {
        assert(posTokens.nonEmpty)

        val ner = TaggedSentence(nerTokens)
        val pos = TaggedSentence(posTokens)

        sentences.append((ner, pos))
        lastSentence.clear()
      }
    }

    def closeDocument = {

      val result = (doc.toString, sentences.toList, docId)
      doc.clear()
      sentences.clear()

      if (result._1.nonEmpty) {
        Some(result._1, result._2, if (includeDocId) docId else None)
      } else
        None
    }

    val docs = lines
      .flatMap { line =>
        val items = line.trim.split(delimiter)
        if (items.nonEmpty && items(0) == "-DOCSTART-") {
          addSentence()
          val closedDoc = closeDocument
          docId = items.lift(2)
          closedDoc
        } else if (items.length <= 1) {
          if (!explodeSentences && (doc.nonEmpty && !doc.endsWith(
              System.lineSeparator) && lastSentence.nonEmpty)) {
            doc.append(System.lineSeparator * 2)
          }
          addSentence()
          if (explodeSentences)
            closeDocument
          else
            None
        } else if (items.length > conllLabelIndex) {
          if (doc.nonEmpty && !doc.endsWith(System.lineSeparator()))
            doc.append(delimiter)

          val begin = doc.length
          doc.append(items(0))
          val end = doc.length - 1
          val tag = items(conllLabelIndex)
          val posTag = items(conllPosIndex)
          val ner = IndexedTaggedWord(items(0), tag, begin, end)
          val pos = IndexedTaggedWord(items(0), posTag, begin, end)
          lastSentence.append((ner, pos))
          None
        } else {
          None
        }
      }

    addSentence()

    val last = if (doc.nonEmpty) Seq((doc.toString, sentences.toList, docId)) else Seq.empty

    (docs ++ last).map {
      case (text, textSentences: Seq[(NerTaggedSentence, PosTaggedSentence)], docId) =>
        val (ner, pos) = textSentences.unzip
        CoNLLDocument(text, ner, pos, docId)
    }
  }

  def packNerTagged(sentences: Seq[NerTaggedSentence]): Seq[Annotation] = {
    NerTagged.pack(sentences)
  }

  def packAssembly(text: String, isTraining: Boolean = true): Seq[Annotation] = {
    new DocumentAssembler()
      .assemble(text, Map("training" -> isTraining.toString))
  }

  def packSentence(text: String, sentences: Seq[TaggedSentence]): Seq[Annotation] = {
    val indexedSentences = sentences.zipWithIndex.map { case (sentence, index) =>
      val start = sentence.indexedTaggedWords.map(t => t.begin).min
      val end = sentence.indexedTaggedWords.map(t => t.end).max
      val sentenceText = text.substring(start, end + 1)
      new Sentence(sentenceText, start, end, index)
    }

    SentenceSplit.pack(indexedSentences)
  }

  def packTokenized(text: String, sentences: Seq[TaggedSentence]): Seq[Annotation] = {
    val tokenizedSentences = sentences.zipWithIndex.map { case (sentence, index) =>
      val tokens = sentence.indexedTaggedWords.map(t => IndexedToken(t.word, t.begin, t.end))
      TokenizedSentence(tokens, index)
    }

    TokenizedWithSentence.pack(tokenizedSentences)
  }

  def packPosTagged(sentences: Seq[TaggedSentence]): Seq[Annotation] = {
    PosTagged.pack(sentences)
  }

  def removeSurroundingHyphens(text: String) =
    "-(.+)-".r.findFirstMatchIn(text).map(_.group(1)).getOrElse(text)

  val annotationType: ArrayType = ArrayType(Annotation.dataType)

  def getAnnotationType(
      column: String,
      annotatorType: String,
      addMetadata: Boolean = true): StructField = {
    if (!addMetadata)
      StructField(column, annotationType, nullable = false)
    else {
      val metadataBuilder: MetadataBuilder = new MetadataBuilder()
      metadataBuilder.putString("annotatorType", annotatorType)
      StructField(column, annotationType, nullable = false, metadataBuilder.build)
    }
  }

  def schema: StructType = {
    val docId = StructField(conllDocIdCol, StringType)
    val text = StructField(conllTextCol, StringType)
    val doc = getAnnotationType(documentCol, AnnotatorType.DOCUMENT)
    val sentence = getAnnotationType(sentenceCol, AnnotatorType.DOCUMENT)
    val token = getAnnotationType(tokenCol, AnnotatorType.TOKEN)
    val pos = getAnnotationType(posCol, AnnotatorType.POS)
    val label = getAnnotationType(labelCol, AnnotatorType.NAMED_ENTITY)

    if (includeDocId)
      StructType(Seq(docId, text, doc, sentence, token, pos, label))
    else
      StructType(Seq(text, doc, sentence, token, pos, label))
  }

  private def coreTransformation(doc: CoNLLDocument) = {
    val text = doc.text
    val labels = packNerTagged(doc.nerTagged)
    val docs = packAssembly(text)
    val sentences = packSentence(text, doc.nerTagged)
    val tokenized = packTokenized(text, doc.nerTagged)
    val posTagged = packPosTagged(doc.posTagged)
    (text, docs, sentences, tokenized, posTagged, labels)
  }

  private def coreTransformationWithDocId(doc: CoNLLDocument) = {
    val docId = removeSurroundingHyphens(doc.docId.getOrElse("X"))
    val (text, docs, sentences, tokenized, posTagged, labels) = coreTransformation(doc)
    (docId, text, docs, sentences, tokenized, posTagged, labels)
  }

  def packDocs(docs: Seq[CoNLLDocument], spark: SparkSession): Dataset[_] = {
    val preDf = if (includeDocId) {
      spark.createDataFrame(docs.map(coreTransformationWithDocId))
    } else {
      spark.createDataFrame(docs.map(coreTransformation))
    }
    spark.createDataFrame(preDf.rdd, schema)
  }

  def readDataset(
      spark: SparkSession,
      path: String,
      readAs: String = ReadAs.TEXT.toString,
      parallelism: Int = 8,
      storageLevel: StorageLevel = StorageLevel.DISK_ONLY): Dataset[_] = {
    if (path.endsWith("*")) {
      val rdd = spark.sparkContext
        .wholeTextFiles(path, minPartitions = parallelism)
        .flatMap { case (_, content) =>
          val lines = content.split(System.lineSeparator)
          readLines(lines)
        }
        .persist(storageLevel)

      val preDf = if (includeDocId) {
        spark.createDataFrame(rdd.map(coreTransformationWithDocId))
      } else {
        spark.createDataFrame(rdd.map(coreTransformation))
      }
      spark.createDataFrame(preDf.rdd, schema)
    } else {
      val er = ExternalResource(path, readAs, Map("format" -> "text"))
      packDocs(readDocs(er), spark)
    }
  }

  def readDatasetFromLines(lines: Array[String], spark: SparkSession): Dataset[_] = {
    packDocs(readLines(lines), spark)
  }
}
