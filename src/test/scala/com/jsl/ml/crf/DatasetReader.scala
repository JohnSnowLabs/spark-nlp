package com.jsl.ml.crf

import java.io.FileInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import scala.collection.mutable.ArrayBuffer
import scala.io.Source


class TextSentenceLabels(val labels: Seq[String])
class TextSentenceAttrs(val words: Seq[TextAttr])
class TextAttr(val attrs: Seq[(String, String)])


object DatasetReader {

  def getSource(file: String): Source = {
    if (file.endsWith(".gz")) {
      val fis = new FileInputStream(file)
      val zis = new GzipCompressorInputStream(fis)
      Source.fromInputStream(zis)
    } else {
      Source.fromFile(file)
    }
  }

  def readWithLabels(file: String, skipLines: Int = 0): Iterator[(TextSentenceLabels, TextSentenceAttrs)] = {
    val lines = getSource(file)
      .getLines()
      .drop(skipLines)

    var labels = new ArrayBuffer[String]()
    var tokens = new ArrayBuffer[TextAttr]()

    def addToResultIfExists(): Option[(TextSentenceLabels, TextSentenceAttrs)] = {
      if (tokens.nonEmpty) {
        val result = (new TextSentenceLabels(labels), new TextSentenceAttrs(tokens))

        labels = new ArrayBuffer[String]()
        tokens = new ArrayBuffer[TextAttr]()
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

        tokens.append(new TextAttr(attrValues))
        labels.append(words.head)
        None
      }
    }
  }

  def encodeDataset(source: Iterator[(TextSentenceLabels, TextSentenceAttrs)]): CrfDataset = {
    val metadata = new DatasetEncoder()

    val instances = source.map{case (textLabels, textSentence) => {
      var prevLabel = metadata.startLabel
      val (labels, features) = textLabels.labels.zip(textSentence.words).map{case (label, word) => {
        val attrs = word.attrs.map(a => a._1 + "=" + a._2)
        val (labelId, features) = metadata.getFeatures(prevLabel, label, attrs, Seq.empty)
        prevLabel = label

        (labelId, features)
      }}.unzip

      (new InstanceLabels(labels), new Instance(features))
    }}.toList

    new CrfDataset(instances, metadata.getMetadata)
  }


  def readAndEncode(file: String, skipLines: Int): CrfDataset = {
    val textDataset = readWithLabels(file, skipLines)

    encodeDataset(textDataset)
  }

  def readAndEncode(file: String, skipLines: Int, metadata: DatasetMetadata): Iterator[(InstanceLabels, Instance)] = {
    val textDataset = readWithLabels(file, skipLines)

    textDataset.map { case (textLabels, textSentence) => {
      val labelIds = textLabels.labels.map(text => metadata.label2Id.getOrElse(text, -1))
      val items = textSentence.words.map{word =>
        val attrIds = word.attrs.flatMap { case (name, value) => {
          val key = name + "=" + value
          metadata.attr2Id.get(key)
        }}

        val attrValues = attrIds.sortBy(id => id).distinct.map(id => (id, 1f)).toArray
        new SparseArray(attrValues)
      }

      (new InstanceLabels(labelIds), new Instance(items))
    }}
  }
}


