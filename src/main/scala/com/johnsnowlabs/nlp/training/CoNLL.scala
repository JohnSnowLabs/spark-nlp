package com.johnsnowlabs.nlp.training

import com.johnsnowlabs.nlp.annotators.common.Annotated.{NerTaggedSentence, PosTaggedSentence}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, DocumentAssembler}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ResourceHelper, ReadAs}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.collection.mutable.ArrayBuffer

case class CoNLLDocument(text: String,
                         nerTagged: Seq[NerTaggedSentence],
                         posTagged: Seq[PosTaggedSentence]
                        )

case class CoNLL(documentCol: String = "document",
                 sentenceCol: String = "sentence",
                 tokenCol: String = "token",
                 posCol: String = "pos",
                 conllLabelIndex: Int = 3,
                 conllPosIndex: Int = 1,
                 conllTextCol: String = "text",
                 labelCol: String = "label",
                 explodeSentences: Boolean = false
                ) {
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

      val result = (doc.toString, sentences.toList)
      doc.clear()
      sentences.clear()

      if (result._1.nonEmpty)
        Some(result._1, result._2)
      else
        None
    }

    val docs = lines
      .flatMap{line =>
        val items = line.trim.split(" ")
        if (items.nonEmpty && items(0) == "-DOCSTART-") {
          addSentence()
          closeDocument
        } else if (items.length <= 1) {
          if (!explodeSentences && (doc.nonEmpty && !doc.endsWith(System.lineSeparator) && lastSentence.nonEmpty)) {
            doc.append(System.lineSeparator * 2)
          }
          addSentence()
          if (explodeSentences)
            closeDocument
          else
            None
        } else if (items.length > conllLabelIndex) {
          if (doc.nonEmpty && !doc.endsWith(System.lineSeparator()))
            doc.append(" ")

          val begin = doc.length
          doc.append(items(0))
          val end = doc.length - 1
          val tag = items(conllLabelIndex)
          val posTag = items(conllPosIndex)
          val ner = IndexedTaggedWord(items(0), tag, begin, end)
          val pos = IndexedTaggedWord(items(0), posTag, begin, end)
          lastSentence.append((ner, pos))
          None
        }
        else {
          None
        }
      }

    addSentence()

    val last = if (doc.nonEmpty) Seq((doc.toString, sentences.toList)) else Seq.empty

    (docs ++ last).map{case(text, textSentences) =>
      val (ner, pos) = textSentences.unzip
      CoNLLDocument(text, ner, pos)
    }
  }

  def packNerTagged(sentences: Seq[NerTaggedSentence]): Seq[Annotation] = {
    NerTagged.pack(sentences)
  }

  def packAssembly(text: String, isTraining: Boolean = true): Seq[Annotation] = {
    new DocumentAssembler()
      .setCleanupMode("shrink_full")
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

  def packTokenized(text: String, sentences: Seq[TaggedSentence]): Seq[Annotation] = {
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

  val annotationType = ArrayType(Annotation.dataType)

  def getAnnotationType(column: String, annotatorType: String, addMetadata: Boolean = true): StructField = {
    if (!addMetadata)
      StructField(column, annotationType, nullable = false)
    else {
      val metadataBuilder: MetadataBuilder = new MetadataBuilder()
      metadataBuilder.putString("annotatorType", annotatorType)
      StructField(column, annotationType, nullable = false, metadataBuilder.build)
    }
  }

  def schema: StructType = {
    val text = StructField(conllTextCol, StringType)
    val doc = getAnnotationType(documentCol, AnnotatorType.DOCUMENT)
    val sentence = getAnnotationType(sentenceCol, AnnotatorType.DOCUMENT)
    val token = getAnnotationType(tokenCol, AnnotatorType.TOKEN)
    val pos = getAnnotationType(posCol, AnnotatorType.POS)
    val label = getAnnotationType(labelCol, AnnotatorType.NAMED_ENTITY)

    StructType(Seq(text, doc, sentence, token, pos, label))
  }

  def packDocs(docs: Seq[CoNLLDocument], spark: SparkSession): Dataset[_] = {
    import spark.implicits._

    val rows = docs.map { doc =>
      val text = doc.text
      val labels = packNerTagged(doc.nerTagged)
      val docs = packAssembly(text)
      val sentences = packSentence(text, doc.nerTagged)
      val tokenized = packTokenized(text, doc.nerTagged)
      val posTagged = packPosTagged(doc.posTagged)

      (text, docs, sentences, tokenized, posTagged, labels)
    }.toDF.rdd

    spark.createDataFrame(rows, schema)
  }

  def readDataset(
                   spark: SparkSession,
                   path: String,
                   readAs: String = ReadAs.LINE_BY_LINE.toString
                 ): Dataset[_] = {
    val er = ExternalResource(path, readAs, Map("format" -> "text"))
    packDocs(readDocs(er), spark)
  }

  def readDatasetFromLines(lines: Array[String], spark: SparkSession): Dataset[_] = {
    packDocs(readLines(lines), spark)
  }
}
