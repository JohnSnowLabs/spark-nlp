package com.johnsnowlabs.nlp.pretrained

import java.io.{FileWriter, InputStream}
import java.sql.Timestamp

import com.johnsnowlabs.nlp.pretrained.ResourceType.ResourceType
import com.johnsnowlabs.util.Version
import org.json4s.NoTypeHints
import org.json4s.ext.EnumNameSerializer
import org.json4s.jackson.JsonMethods.parse
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.write

import scala.io.Source


case class ResourceMetadata
(
  name: String,
  language: Option[String],
  libVersion: Option[Version],
  sparkVersion: Option[Version],
  readyToUse: Boolean,
  time: Timestamp,
  isZipped: Boolean = false,
  category: Option[ResourceType] = Some(ResourceType.NOT_DEFINED),
  checksum: String = ""
) {


  lazy val key = {
    s"${name}_${s(language)}_${v(libVersion)}_${v(sparkVersion)}_${t(time)}"
  }

  lazy val fileName = {
    if (isZipped) key + ".zip" else key
  }

  private def s(str: Option[String]): String = {
    str.getOrElse("")
  }

  private def v(ver: Option[Version]): String = {
    ver.map(v => v.toString).getOrElse("")
  }

  private def t(time: Timestamp): String = {
    time.getTime.toString
  }
}


object ResourceMetadata {
  implicit val formats = Serialization.formats(NoTypeHints) + new EnumNameSerializer(ResourceType)

  def toJson(meta: ResourceMetadata): String  = {
    write(meta)
  }

  def parseJson(json: String): ResourceMetadata = {
    val parsed = parse(json)
    parsed.extract[ResourceMetadata]
  }

  def resolveResource(candidates: List[ResourceMetadata],
                      request: ResourceRequest): Option[ResourceMetadata] = {

    candidates
      .filter(item => item.readyToUse
        && item.name == request.name
        && (request.language.isEmpty || item.language.isEmpty || request.language.get == item.language.get)
        && Version.isCompatible(request.libVersion, item.libVersion)
        && Version.isCompatible(request.sparkVersion, item.sparkVersion)
      )
      .sortBy(item => item.time.getTime)
      .lastOption
  }

  def readResources(file: String): List[ResourceMetadata] = {
    readResources(Source.fromFile(file))
  }

  def readResources(stream: InputStream): List[ResourceMetadata] = {
    readResources(Source.fromInputStream(stream))
  }

  def readResources(source: Source): List[ResourceMetadata] = {
    source.getLines()
      .collect{case line if line.nonEmpty =>
        ResourceMetadata.parseJson(line)
      }
      .toList
  }

  def addMetadataToFile(fileName: String, metadata: ResourceMetadata): Unit = {
    val fw = new FileWriter(fileName, true)
    try {
      fw.write("\n" + ResourceMetadata.toJson(metadata))
    }
    finally fw.close()
  }
}
