/*
 * Copyright 2017-2025 John Snow Labs
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

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.nlp.annotators.common.DatasetHelpers._
import com.johnsnowlabs.nlp.annotators.common.NerTagged.{getAnnotations, getLabelsFromSentences}
import com.johnsnowlabs.nlp.annotators.common.WordpieceEmbeddingsSentence
import org.apache.spark.sql.{Dataset, Row}
import org.slf4j.LoggerFactory

import java.util.concurrent.{ExecutorService, Executors, LinkedBlockingQueue, TimeUnit}
import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.ExecutionContext
import scala.jdk.CollectionConverters._

/** Configuration for the NerDLDataLoader.
  *
  * @param batchSize
  *   Number of sentences per batch (default: 16)
  * @param prefetchBatches
  *   Number of batches to prefetch per worker (default: 2). Total prefetch buffer size will be
  *   numWorkers * prefetchFactor
  * @param shuffleInPartition
  *   Whether to shuffle the data (default: true). Improves training convergence.
  * @param timeoutMillis
  *   Timeout in milliseconds for fetching a batch (default: 10000). Prevents hanging on slow
  *   operations.
  */
case class DataLoaderConfig(
    batchSize: Int = 16,
    prefetchBatches: Int = 20,
    shuffleInPartition: Boolean = true,
    timeoutMillis: Long = 10000)

/** DataLoader for NerDLApproach with threaded prefetching.
  *
  * This class provides an efficient way to load training data for NER models by:
  *   - Prefetching batches in background threads to overlap I/O with computation
  *   - Using a bounded queue to prevent excessive memory usage
  *
  * @param config
  *   Configuration for the data loader
  */
class NerDLDataLoader(config: DataLoaderConfig = DataLoaderConfig()) {
  import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._

  @volatile private var isShutdown = false
  private var executorService: Option[ExecutorService] = None
  private var executionContext: Option[ExecutionContext] = None
  private val logger = LoggerFactory.getLogger(this.getClass)

  /** Creates an iterator that prefetches and yields batches of NER training data from a Spark
    * DataFrame.
    *
    * The iterator uses background threads to prefetch batches while the main thread consumes
    * them, improving throughput by overlapping I/O with computation.
    *
    * @param dataset
    *   Spark DataFrame containing the training data
    * @param inputCols
    *   TOKEN and EMBEDDING type input columns
    * @param labelColumn
    *   Column name containing the NER labels
    * @return
    *   Iterator over batches, where each batch is an Array of (labels, embeddings) pairs
    */
  def createIterator(
      dataset: Dataset[Row],
      inputCols: Seq[String],
      labelColumn: String): Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] = {
    if (config.prefetchBatches <= 0) {
      // Single-threaded mode - no prefetching and directly consume from the dataset
      createSourceBatchIterator(dataset, inputCols, labelColumn, config.batchSize)
    } else {
      // Threaded mode with prefetching
      createThreadedIterator(dataset, inputCols, labelColumn)
    }
  }

  /** Creates an iterator over batches of NER training data directly from a Spark DataFrame using
    * `toLocalIterator`.
    *
    * We should probably take from this iterator as much as possible (within RAM limits) to
    * trigger partition computation across the cluster.
    *
    * @param dataset
    *   Spark DataFrame containing the training data
    * @param inputCols
    *   TOKEN and EMBEDDING type input columns
    * @param labelColumn
    *   Column name containing the NER labels
    * @param batchSize
    *   Number of sentences per batch
    * @return
    *   Iterator over batches, where each batch is an Array of (labels, embeddings) pairs
    */
  private def createSourceBatchIterator(
      dataset: Dataset[Row],
      inputCols: Seq[String],
      labelColumn: String,
      batchSize: Int): Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] = {

    def processPartition(
        it: Iterator[Row]): Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] =
      new Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] {
        // create a batch
        override def next(): Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)] = {
          var count = 0
          val thisBatch = new ArrayBuffer[(TextSentenceLabels, WordpieceEmbeddingsSentence)]

          while (it.hasNext && count < batchSize) {
            count += 1
            val nextRow = it.next

            val labelAnnotations = getAnnotations(nextRow, labelColumn)
            val sentenceAnnotations =
              inputCols.flatMap(s => getAnnotations(nextRow, s))
            val sentences = WordpieceEmbeddingsSentence.unpack(sentenceAnnotations)
            val labels = getLabelsFromSentences(sentences, labelAnnotations)
            val thisOne = labels.zip(sentences)

            thisBatch ++= thisOne
          }
          thisBatch.toArray
        }

        override def hasNext: Boolean = it.hasNext
      }

    // Process each partition on worker nodes
    val selected = dataset.select(labelColumn, inputCols: _*)
    (
      // to improve training
      // NOTE: This might have implications on model performance, partitions themselves are not shuffled
      if (config.shuffleInPartition) selected.randomize
      else
        selected
    )
      .mapPartitions(processPartition) // create batches in each partition
      .as[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]]
      .toLocalIterator()
      .asScala
  }

  /** Worker runnable that loads batches and puts them in the queue.
    *
    * @param batchQueue
    *   Blocking queue to hold loaded batches
    * @param sourceIterator
    *   Iterator over source batches
    */
  private class BatchLoaderThread(
      batchQueue: LinkedBlockingQueue[
        Option[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]]],
      sourceIterator: Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]])
      extends Runnable {
    private val logger = LoggerFactory.getLogger(this.getClass)

    override def run(): Unit = {
      try {
        while (!isShutdown && sourceIterator.hasNext) {
          // Fetch the next batch
          val batch = sourceIterator.next()

          // Offer to queue (blocking with timeout)
          var offered = false
          while (!offered && !isShutdown) {
            offered = batchQueue.offer(Some(batch), config.timeoutMillis, TimeUnit.MILLISECONDS)
          }
        }
      } catch {
        case _: InterruptedException =>
          Thread.currentThread().interrupt()
        case e: Exception =>
          logger.error(s"Fetcher Error: ${e.getMessage}")
          e.printStackTrace()
      } finally {
        // Sentinel: Signal end of data
        // Either due to completion or shutdown
        try {
          batchQueue.put(None)
        } catch {
          case _: InterruptedException => // Ignore during shutdown
        }
      }
    }

  }

  private def createThreadedIterator(
      dataset: Dataset[Row],
      sentenceCols: Seq[String],
      labelColumn: String): Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] = {

    // Queue Capacity: holds completed batches.
    val queueCapacity = config.prefetchBatches
    val batchQueue =
      new LinkedBlockingQueue[Option[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]]](
        queueCapacity)

    // Source data iterator
    val sourceBatchIterator =
      createSourceBatchIterator(dataset, sentenceCols, labelColumn, config.batchSize)

    // Create a producer thread for prefetching.
    val executor = Executors.newSingleThreadExecutor()
    executorService = Some(executor)
    logger.info(s"Starting data loader thread with prefetch buffer size: $queueCapacity batches.")
    executor.submit(new BatchLoaderThread(batchQueue, sourceBatchIterator))

    // Consumer Iterator (Main Thread)
    new BatchLoaderIterator(batchQueue)
  }

  private class BatchLoaderIterator(
      batchQueue: LinkedBlockingQueue[
        Option[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]]])
      extends Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] {
    private var nextBatch: Option[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] =
      None
    private var endOfData = false

    @tailrec
    final override def hasNext: Boolean = {
      if (endOfData) false
      else if (nextBatch.isDefined) true // Already have a batch ready
      else {
        // Poll from queue in advance (to avoid blocking in next())
        val result = batchQueue.poll(config.timeoutMillis, TimeUnit.MILLISECONDS)

        result match {
          case null =>
            // Timeout waiting for Spark. Training is faster than Data Loading.
            // Wait for next batch.
            if (isShutdown) false
            else {
              hasNext
            }
          case None =>
            endOfData = true // Signal: No more data
            false
          case Some(batch) =>
            nextBatch = Some(batch)
            true
        }
      }
    }

    override def next(): Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)] = {
      if (!hasNext) throw new NoSuchElementException("No more batches")
      val batch = nextBatch.get
      nextBatch = None
      batch
    }
  }

  /** Shuts down the data loader and releases all resources.
    *
    * This method should be called when there is still data but the loader is no longer needed to
    * prevent resource leaks. It's safe to call multiple times.
    */
  def shutdown(): Unit = {
    isShutdown = true
    executorService.foreach { executor =>
      executor.shutdown()
      try {
        if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
          executor.shutdownNow()
        }
      } catch {
        case _: InterruptedException =>
          executor.shutdownNow()
          Thread.currentThread().interrupt()
      }
    }
    executorService = None
    executionContext = None
  }
}

/** Companion object providing factory methods for NerDLDataLoader. */
object NerDLDataLoader {
  def iterateOnDataframe(
      dataset: Dataset[Row],
      inputColumns: Array[String],
      labelColumn: String,
      batchSize: Int,
      prefetchBatches: Int,
      shuffleInPartition: Boolean = true)
      : Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] = {
    new NerDLDataLoader(
      DataLoaderConfig(
        batchSize = batchSize,
        prefetchBatches = prefetchBatches,
        shuffleInPartition = shuffleInPartition))
      .createIterator(dataset, inputColumns, labelColumn)
  }
}
