package com.johnsnowlabs.util

import java.io.File
import java.nio.file.Paths
import java.sql.Timestamp
import java.util.Date

import com.johnsnowlabs.nlp.pretrained.{ResourceDownloader, ResourceMetadata}
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.util.MLWriter


object TrainingHelper {

  def saveModel(name: String,
                language: Option[String],
                libVersion: Option[Version],
                sparkVersion: Option[Version],
                category: Option[String] = Some(ResourceDownloader.NOT_DEFINED),
                modelWriter: MLWriter,
                folder: String
               ): Unit = {

    // 1. Get current timestamp
    val timestamp = new Timestamp(new Date().getTime)

    // 2. Create resource metadata
    val meta = new ResourceMetadata(name, language, libVersion, sparkVersion, true, timestamp, true, category = category)

    // 3. Save model to file
    val file = Paths.get(folder, meta.key).toString.replaceAllLiterally("\\", "/")
    modelWriter.save(file)

    // 4. Zip file
    val zipFile = Paths.get(folder, meta.fileName).toString
    ZipArchiveUtil.zip(file, zipFile)

    // 5. Remove original file
    try {
      FileUtils.deleteDirectory(new File(file))
    } catch {
      case _: java.io.IOException => //file lock may prevent deletion, ignore and continue
    }

      // 6. Add to metadata.json info about resource
      val metadataFile = Paths.get(folder, "metadata.json").toString
      ResourceMetadata.addMetadataToFile(metadataFile, meta)
    }
}
