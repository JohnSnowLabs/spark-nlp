package com.johnsnowlabs.storage

import java.io.File
import org.apache.spark.SparkFiles

/*
  1. Copy Embeddings to local tmp file
  2. Index Embeddings if need
  3. Copy Index to cluster
  4. Open RocksDb based Embeddings on local index (lazy)
 */
abstract class StorageConnection[A, +B <: RocksDBRetriever[A]](val fileName: String, val caseSensitive: Boolean) extends Serializable {

  @transient private lazy val retriever: B = createRetriever(fileName, caseSensitive)

  protected def createRetriever(localPath: String, caseSensitive: Boolean): B

  def getLocalRetriever: B = retriever

//  def getLocalRetriever: B = {
//    val localPath = StorageHelper.getLocalPath(fileName)
//    if (Option(retriever).isDefined)
//      retriever
//    else if (new File(localPath).exists()) {
//      retriever = createRetriever(localPath, caseSensitive)
//      retriever
//    }
//    else {
//      val localFromClusterPath = SparkFiles.get(fileName)
//      require(new File(localFromClusterPath).exists(), s"Storage not found under given ref: ${fileName.replaceAll("/storage_", "")}\n" +
//        s" This usually means:\n1. You have not loaded any storage under such ref\n2." +
//        s" You are trying to use cluster mode without a proper shared filesystem.\n3. source was not provided to Storage creation" +
//        s"\n4. If you are trying to utilize Storage defined elsewhere, make sure you it's appropriate ref. ")
//      retriever = createRetriever(localFromClusterPath, caseSensitive)
//      retriever
//    }
//  }

}