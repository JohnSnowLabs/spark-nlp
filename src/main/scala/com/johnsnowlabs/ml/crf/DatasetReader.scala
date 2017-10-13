package com.johnsnowlabs.ml.crf

import java.io.FileInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import scala.collection.mutable.ArrayBuffer
import scala.io.Source


case class TextSentenceLabels(labels: Seq[String])
case class TextSentenceAttrs(words: Seq[WordAttrs])
case class WordAttrs(attrs: Seq[(String, String)])


object DatasetReader {

  private def getSource(file: String): Source = {
    if (file.endsWith(".gz")) {
      val fis = new FileInputStream(file)
      val zis = new GzipCompressorInputStream(fis)
      Source.fromInputStream(zis)
    } else {
      Source.fromFile(file)
    }
  }

  private def readWithLabels(file: String, skipLines: Int = 0): Iterator[(TextSentenceLabels, TextSentenceAttrs)] = {
    val lines = getSource(file)
      .getLines()
      .drop(skipLines)

    var labels = new ArrayBuffer[String]()
    var tokens = new ArrayBuffer[WordAttrs]()

    def addToResultIfExists(): Option[(TextSentenceLabels, TextSentenceAttrs)] = {
      if (tokens.nonEmpty) {
        val result = (TextSentenceLabels(labels), TextSentenceAttrs(tokens))

        labels = new ArrayBuffer[String]()
        tokens = new ArrayBuffer[WordAttrs]()
        Some(result)
      }
      else {
        None
      }
    }

    lines.flatMap{line =>
      val words = line.split("\t")
      if (words.length <= 1) {
        addToResultIfExists()
      } else {
        val attrValues = words
          .drop(1)
          .map(feature => {
            val attrValue = feature.split("=")
            val attr = attrValue(0)
            val value = if (attrValue.size == 1) "" else attrValue(1)

            (attr, value)
          })

        tokens.append(WordAttrs(attrValues))
        labels.append(words.head)
        None
      }
    }
  }

  def encodeDataset(source: Iterator[(TextSentenceLabels, TextSentenceAttrs)]): CrfDataset = {
    val metadata = new DatasetEncoder()

    val instances = source.map{case (textLabels, textSentence) =>
      var prevLabel = metadata.startLabel
      val (labels, features) = textLabels.labels.zip(textSentence.words)
        .map{case (label, word) =>
          val attrs = word.attrs.map(a => a._1 + "=" + a._2)
          val (labelId, features) = metadata.getFeatures(prevLabel, label, attrs, Seq.empty)
          prevLabel = label

          (labelId, features)
        }.unzip

      (InstanceLabels(labels), Instance(features))
    }.toList

    CrfDataset(instances, metadata.getMetadata)
  }

  private def encodeLabels(labels: TextSentenceLabels, metadata: DatasetMetadata): InstanceLabels = {
    val labelIds = labels.labels.map(text => metadata.label2Id.getOrElse(text, -1))
    InstanceLabels(labelIds)
  }

  def encodeSentence(sentence: TextSentenceAttrs, metadata: DatasetMetadata): Instance = {
    val items = sentence.words.map{word =>
      val attrIds = word.attrs.flatMap { case (name, value) =>
        val key = name + "=" + value
        metadata.attr2Id.get(key)
      }

      val attrValues = attrIds.sortBy(id => id).distinct.map(id => (id, 1f)).toArray
      new SparseArray(attrValues)
    }

    Instance(items)
  }

  def readAndEncode(file: String, skipLines: Int): CrfDataset = {
    val textDataset = readWithLabels(file, skipLines)

    encodeDataset(textDataset)
  }

  def readAndEncode(file: String, skipLines: Int, metadata: DatasetMetadata): Iterator[(InstanceLabels, Instance)] = {
    val textDataset = readWithLabels(file, skipLines)

    textDataset.map{case (sourceLabels, sourceInstance) =>
      val labels = encodeLabels(sourceLabels, metadata)
      val instance = encodeSentence(sourceInstance, metadata)
      (labels, instance)
    }
  }
}


