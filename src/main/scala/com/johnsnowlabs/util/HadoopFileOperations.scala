package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import org.apache.hadoop.fs.Path

object HadoopFileOperations {

  val fs = ResourceDownloader.fileSystem

  def moveFile(localFile: String, hadoopFile: String) {
    fs.copyFromLocalFile(new Path(localFile), new Path(hadoopFile))
  }

}
