package com.johnsnowlabs.nlp.datasets

import com.johnsnowlabs.nlp.annotators.common.Annotated.{NerTaggedSentence, PosTaggedSentence}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, DocumentAssembler}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ResourceHelper}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.collection.mutable.ArrayBuffer

case class CoNLLDocument(text: String,
                         nerTagged: Seq[NerTaggedSentence],
                         posTagged: Seq[PosTaggedSentence]
                        )

case class CoNLL(targetColumn: Int = 3,
                 posColumn: Int = 1,
                 textColumn: String = "text",
                 docColumn: String = "document",
                 sentenceColumn: String = "sentence",
                 tokenColumn: String = "token",
                 posTaggedColumn: String = "pos",
                 labelColumn: String = "label"
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
    val lastSentence = new ArrayBuffer[(IndexedTaggedWord, IndexedTaggedWord)]()

    val sentences = new ArrayBuffer[(TaggedSentence, TaggedSentence)]()

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

    val docs = lines
      .flatMap{line =>
        val items = line.trim.split(" ")
        if (items.nonEmpty && items(0) == "-DOCSTART-") {
          addSentence()

          val result = (doc.toString, sentences.toList)
          doc.clear()
          sentences.clear()

          if (result._1.nonEmpty)
            Some(result._1, result._2)
          else
            None
        } else if (items.length <= 1) {
          if (doc.nonEmpty && !doc.endsWith(System.lineSeparator) && lastSentence.nonEmpty) {
            doc.append(System.lineSeparator * 2)
          }
          addSentence()
          None
        } else if (items.length > targetColumn) {
          if (doc.nonEmpty && !doc.endsWith(System.lineSeparator()))
            doc.append(" ")

          val begin = doc.length
          doc.append(items(0))
          val end = doc.length - 1
          val tag = items(targetColumn)
          val posTag = items(posColumn)
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

    (docs ++ last).map{case(text, sentences) =>
      val (ner, pos) = sentences.unzip
      CoNLLDocument(text, ner, pos)
    }
  }

  def packNerTagged(sentences: Seq[NerTaggedSentence]): Seq[Annotation] = {
    NerTagged.pack(sentences)
  }

  def packAssembly(text: String, isTraining: Boolean = true): Seq[Annotation] = {
    new DocumentAssembler().assemble(text, Map("training" -> isTraining.toString))
  }

  def packSentence(text: String, sentences: Seq[TaggedSentence]): Seq[Annotation] = {
    val indexedSentences = sentences.map{sentence =>
      val start = sentence.indexedTaggedWords.map(t => t.begin).min
      val end = sentence.indexedTaggedWords.map(t => t.end).max
      val sentenceText = text.substring(start, end + 1)
      new Sentence(sentenceText, start, end)}

    SentenceSplit.pack(indexedSentences)
  }

  def packTokenized(text: String, sentences: Seq[TaggedSentence]): Seq[Annotation] = {
    val tokenizedSentences = sentences.map{sentence =>
      val tokens = sentence.indexedTaggedWords.map(t =>
        IndexedToken(t.word, t.begin, t.end)
      )
      TokenizedSentence(tokens)
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

  def schema(): StructType = {
    val text = StructField(textColumn, StringType)
    val doc = getAnnotationType(docColumn, AnnotatorType.DOCUMENT)
    val sentence = getAnnotationType(sentenceColumn, AnnotatorType.DOCUMENT)
    val token = getAnnotationType(tokenColumn, AnnotatorType.TOKEN)
    val pos = getAnnotationType(posTaggedColumn, AnnotatorType.POS)
    val label = getAnnotationType(labelColumn, AnnotatorType.NAMED_ENTITY)

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

  def readDataset(er: ExternalResource,
                  spark: SparkSession
                 ): Dataset[_] = {

    packDocs(readDocs(er), spark)
  }

  def readDatasetFromLines(lines: Array[String], spark: SparkSession): Dataset[_] = {
    packDocs(readLines(lines), spark)
  }
}
