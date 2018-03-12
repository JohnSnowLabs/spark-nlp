package com.johnsnowlabs.ml.tensorflow

import java.nio.LongBuffer
import java.nio.file.{Files, Paths}

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.nlp.annotators.common.TokenizedSentence
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import org.slf4j.LoggerFactory
import org.tensorflow.{Graph, Session, Tensor}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


class TensorflowNer
(
  val tensorflow: TensorflowWrapper,
  val encoder: DatasetEncoder,
  val batchSize: Int,
  val verbose: Verbose.Value
) {

  private val logger = LoggerFactory.getLogger("NerDL")

  private def log(value: => String, minLevel: Verbose.Level): Unit = {
    if (minLevel.id >= verbose.id) {
      logger.info(value)
    }
  }

  private val charIdsKey = "char_repr/char_ids"
  private val wordLengthsKey = "char_repr/word_lengths"
  private val wordEmbeddingsKey = "word_repr_1/word_embeddings"
  private val sentenceLengthsKey = "word_repr/sentence_lengths"
  private val dropoutKey = "training/dropout"

  private val learningRateKey = "training/lr"
  private val labelsKey = "training/labels"

  private val lossKey = "inference/loss"
  private val trainingKey = "training_1/Momentum"
  private val predictKey = "context_repr/predicted_labels"


  private def extractInts(source: Tensor[_], size: Int): Array[Int] = {
    val buffer = LongBuffer.allocate(size)
    source.writeTo(buffer)
    buffer.array().map(item => item.toInt)
  }

  def predict(dataset: Array[TokenizedSentence]): Array[Array[String]] = {

    val result = ArrayBuffer[Array[String]]()

    for (slice <- dataset.grouped(batchSize)) {
      val sentences = slice.map(r => r.tokens)

      val batchInput = encoder.encodeInputData(sentences)

      val tensors = new TensorResources()

      val calculated = tensorflow.session.runner
        .feed(sentenceLengthsKey, tensors.createTensor(batchInput.sentenceLengths))
        .feed(wordEmbeddingsKey, tensors.createTensor(batchInput.wordEmbeddings))

        .feed(wordLengthsKey, tensors.createTensor(batchInput.wordLengths))
        .feed(charIdsKey, tensors.createTensor(batchInput.charIds))

        .feed(dropoutKey, tensors.createTensor(1.1f))
        .fetch(predictKey)
        .run()

      tensors.clearTensors()

      val tagIds = extractInts(calculated.get(0), batchSize * batchInput.maxLength)
      val tags = encoder.decodeOutputData(tagIds)
      val sentenceTags = encoder.convertBatchTags(tags, batchInput.sentenceLengths)

      result.appendAll(sentenceTags)
    }

    result.toArray
  }

  def train(trainDataset: Array[(TextSentenceLabels, TokenizedSentence)],
            lr: Float,
            po: Float,
            batchSize: Int,
            dropout: Float,
            startEpoch: Int,
            endEpoch: Int
           ): Unit = {

    log(s"Training started, trainExamples: ${trainDataset.length}, " +
      s"labels: ${encoder.tags.length} " +
      s"chars: ${encoder.chars.length}, ", Verbose.TrainingStat)

    // Initialize
    if (startEpoch == 0)
      tensorflow.session.runner.addTarget("training_1/init").run()

    // Train
    for (epoch <- startEpoch until endEpoch) {

      val epochDataset = Random.shuffle(trainDataset.toList).toArray
      val learningRate = lr / (1 + po * epoch)

      log(s"Epoch: $epoch started, learning rate: $learningRate, dataset size: ${epochDataset.length}", Verbose.Epochs)

      val time = System.nanoTime()
      var batches = 0
      var loss = 0f
      for (slice <- epochDataset.grouped(batchSize)) {
        val sentences = slice.map(r => r._2.tokens)
        val tags = slice.map(r => r._1.labels.toArray)

        val batchInput = encoder.encodeInputData(sentences)
        val batchTags = encoder.encodeTags(tags)

        val tensors = new TensorResources()
        val calculated = tensorflow.session.runner
          .feed(sentenceLengthsKey, tensors.createTensor(batchInput.sentenceLengths))
          .feed(wordEmbeddingsKey, tensors.createTensor(batchInput.wordEmbeddings))

          .feed(wordLengthsKey, tensors.createTensor(batchInput.wordLengths))
          .feed(charIdsKey, tensors.createTensor(batchInput.charIds))
          .feed(labelsKey, tensors.createTensor(batchTags))

          .feed(dropoutKey, tensors.createTensor(dropout))
          .feed(learningRateKey, tensors.createTensor(learningRate))

          .fetch(lossKey)
          .addTarget(trainingKey)
          .run()

        loss += calculated.get(0).floatValue()

        tensors.clearTensors()
        batches += 1
      }

      System.out.println(s"Done, ${(System.nanoTime() - time)/1e9} loss: $loss, batches: $batches")
    }
  }
}


object TensorflowNer {

  def apply(encoder: DatasetEncoder, batchSize: Int, verbose: Verbose.Value) = {
    val graph = new Graph()
    val session = new Session(graph)
    graph.importGraphDef(Files.readAllBytes(Paths.get("char_cnn_blstm_30_25_100_200.pb")))

    val tf = new TensorflowWrapper(session, graph)

    new TensorflowNer(tf, encoder, batchSize, verbose)
  }
}


// ToDo Delete after merge
import java.io.{File, FileInputStream, FileOutputStream, IOException}
import java.util.zip.{ZipEntry, ZipFile, ZipOutputStream}

import org.apache.commons.io.FileUtils

import scala.collection.JavaConversions.enumerationAsScalaIterator
import scala.io.{BufferedSource, Codec}

/**
  * Copied from https://github.com/dhbikoff/Scala-Zip-Archive-Util
  * with small fixes
  */
object ZipArchiveUtil {

  private def listFiles(file: File, outputFilename: String): List[String] = {
    file match {
      case file if file.isFile => {
        if (file.getName != outputFilename)
          List(file.getAbsoluteFile.toString)
        else
          List()
      }
      case file if file.isDirectory => {
        val fList = file.list
        // Add all files in current dir to list and recur on subdirs
        fList.foldLeft(List[String]())((pList: List[String], path: String) =>
          pList ++ listFiles(new File(file, path), outputFilename))
      }
      case _ => throw new IOException("Bad path. No file or directory found.")
    }
  }

  private def addFileToZipEntry(filename: String, parentPath: String,
                                filePathsCount: Int): ZipEntry = {
    if (filePathsCount <= 1)
      new ZipEntry(new File(filename).getName)
    else {
      // use relative path to avoid adding absolute path directories
      val relative = new File(parentPath).toURI.
        relativize(new File(filename).toURI).getPath
      new ZipEntry(relative)
    }
  }

  private def createZip(filePaths: List[String], outputFilename: String, parentPath: String) = {
    try {
      val fileOutputStream = new FileOutputStream(outputFilename)
      val zipOutputStream = new ZipOutputStream(fileOutputStream)

      filePaths.foreach((name: String) => {
        val zipEntry = addFileToZipEntry(name, parentPath, filePaths.size)
        zipOutputStream.putNextEntry(zipEntry)
        val inputSrc = new BufferedSource(
          new FileInputStream(name))(Codec.ISO8859)
        inputSrc foreach { c: Char => zipOutputStream.write(c) }
        inputSrc.close
      })

      zipOutputStream.closeEntry
      zipOutputStream.close
      fileOutputStream.close

    } catch {
      case e: IOException => {
        e.printStackTrace
      }
    }
  }

  def zip(fileName: String, outputFileName: String): Unit = {
    val file = new File(fileName)
    val filePaths = listFiles(file, outputFileName)
    createZip(filePaths, outputFileName, fileName)
  }

  def unzip(file: File, destDirPath: Option[String] = None): String = {
    val basename = file.getName.substring(0, file.getName.lastIndexOf("."))

    val destDir = if (destDirPath == None) {
      new File(file.getParentFile, basename)
    }
    else {
      new File(destDirPath.get)
    }

    destDir.mkdirs()

    val zip = new ZipFile(file)
    zip.entries foreach { entry =>
      val entryName = entry.getName
      val entryPath = {
        if (entryName.startsWith(basename))
          entryName.substring(basename.length)
        else
          entryName
      }

      // create output directory if it doesn't exist already
      val splitPath = entryName.split(File.separator.replace("\\", "/")).dropRight(1)

      val dirBuilder = new StringBuilder(destDir.getPath)
      for (part <- splitPath) {
        dirBuilder.append(File.separator)
        dirBuilder.append(part)
        val path = dirBuilder.toString

        if (!(new File(path).exists)) {
          new File(path).mkdir
        }
      }

      // write file to dest
      FileUtils.copyInputStreamToFile(zip.getInputStream(entry),
        new File(destDir, entryPath))
    }

    destDir.getPath
  }
}