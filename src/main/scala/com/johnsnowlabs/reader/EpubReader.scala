/*
 * Copyright 2017-2026 John Snow Labs
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
package com.johnsnowlabs.reader

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.partition.util.PartitionHelper.datasetWithBinaryFile
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import java.nio.charset.{Charset, StandardCharsets}
import java.util.Base64
import java.util.zip.ZipInputStream
import scala.collection.mutable
import scala.xml.XML

/** Class to read and parse EPUB files.
  *
  * EPUB content is extracted by resolving the package manifest and spine, then parsing the XHTML
  * chapters with the existing HTML reader to preserve the same text/table heuristics. Embedded
  * images are resolved from the archive and attached as binary payloads so image pipelines can
  * consume them without relying on external paths.
  */
class EpubReader(storeContent: Boolean = false, outputFormat: String = "plain-text")
    extends Serializable {

  private lazy val spark = ResourceHelper.spark

  private var outputColumn = "epub"

  def setOutputColumn(value: String): this.type = {
    require(value.nonEmpty, "Output column name cannot be empty.")
    outputColumn = value
    this
  }

  def getOutputColumn: String = outputColumn

  def epub(filePath: String): DataFrame = {
    if (ResourceHelper.validFile(filePath)) {
      val epubDf = datasetWithBinaryFile(spark, filePath)
        .withColumn(outputColumn, parseEpubUDF(col("content")))
      if (storeContent) epubDf.select("path", outputColumn, "content")
      else epubDf.select("path", outputColumn)
    } else throw new IllegalArgumentException(s"Invalid filePath: $filePath")
  }

  def epubToHTMLElement(content: Array[Byte]): Seq[HTMLElement] = {
    parseEpub(content)
  }

  private case class ManifestItem(
      id: String,
      path: String,
      mediaType: String,
      properties: Set[String] = Set.empty)

  private val parseEpubUDF = udf((data: Array[Byte]) => parseEpub(data))

  private val htmlMediaTypes = Set("application/xhtml+xml", "text/html")

  private def parseEpub(content: Array[Byte]): Seq[HTMLElement] = {
    try {
      val entries = unzipEntries(content)
      val opfPath = findOpfPath(entries)
      val opfBytes = entries.getOrElse(
        opfPath,
        throw new IllegalArgumentException(s"Missing package document: $opfPath"))
      val opfXml = XML.loadString(decodeDocument(opfBytes))
      val manifestItems = parseManifest(opfXml, opfPath)
      val manifestById = manifestItems.map(item => item.id -> item).toMap
      val manifestByPath = manifestItems.map(item => normalizePath(item.path) -> item).toMap
      val chapterPaths = resolveSpine(opfXml, manifestItems, manifestById, entries.keys.toSeq)
      val htmlReader = new HTMLReader(outputFormat = outputFormat)

      var pageOffset = 0

      chapterPaths.zipWithIndex.flatMap { case (chapterPath, sectionIndex) =>
        entries.get(normalizePath(chapterPath)).toSeq.flatMap { chapterBytes =>
          val chapterHtml = decodeDocument(chapterBytes)
          val chapterElements = htmlReader.htmlToHTMLElement(chapterHtml).toSeq
          val chapterPageNumbers =
            chapterElements.flatMap(_.metadata.get("pageNumber").flatMap(parseInt))
          val localMaxPage = if (chapterPageNumbers.nonEmpty) chapterPageNumbers.max else 1
          val elements = chapterElements.map { element =>
            val metadata = mutable.Map.empty[String, String] ++ element.metadata
            val localPage = metadata.get("pageNumber").flatMap(parseInt).getOrElse(1)
            metadata("pageNumber") = (pageOffset + localPage).toString
            metadata("sectionNumber") = (sectionIndex + 1).toString
            metadata("sectionPath") = chapterPath

            if (element.elementType == ElementType.IMAGE) {
              resolveImageElement(element, chapterPath, metadata, manifestByPath, entries)
            } else {
              element.copy(metadata = metadata)
            }
          }

          pageOffset += localMaxPage
          elements
        }
      }
    } catch {
      case e: Exception =>
        Seq(
          HTMLElement(
            ElementType.ERROR,
            s"Could not parse EPUB: ${e.getMessage}",
            mutable.Map.empty[String, String]))
    }
  }

  private def resolveImageElement(
      element: HTMLElement,
      chapterPath: String,
      metadata: mutable.Map[String, String],
      manifestByPath: Map[String, ManifestItem],
      entries: Map[String, Array[Byte]]): HTMLElement = {

    val source = element.content.trim
    metadata("sourcePath") = source

    val resolvedBytes = if (metadata.get("encoding").exists(_.toLowerCase.contains("base64"))) {
      decodeBase64Image(source)
    } else {
      resolveArchiveEntry(chapterPath, source, entries)
    }

    val mediaType = if (metadata.get("encoding").exists(_.toLowerCase.contains("base64"))) {
      metadata
        .get("encoding")
        .flatMap(parseDataUriMediaType)
        .orElse(Some(inferMediaType(source)))
    } else {
      val resolvedPath = resolveRelativePath(chapterPath, source)
      manifestByPath
        .get(normalizePath(resolvedPath))
        .map(_.mediaType)
        .orElse(Some(inferMediaType(source)))
    }

    mediaType.foreach(mt => metadata("mediaType") = mt)

    resolvedBytes match {
      case Some(bytes) =>
        element.copy(content = "", metadata = metadata, binaryContent = Some(bytes))
      case None =>
        element.copy(metadata = metadata)
    }
  }

  private def unzipEntries(content: Array[Byte]): Map[String, Array[Byte]] = {
    val zipInputStream = new ZipInputStream(new ByteArrayInputStream(content))
    val entries = mutable.Map.empty[String, Array[Byte]]

    try {
      Iterator
        .continually(zipInputStream.getNextEntry)
        .takeWhile(_ != null)
        .foreach { entry =>
          val output = new ByteArrayOutputStream()
          val buffer = new Array[Byte](8192)
          Iterator
            .continually(zipInputStream.read(buffer))
            .takeWhile(_ != -1)
            .foreach(read => output.write(buffer, 0, read))
          entries += normalizePath(entry.getName) -> output.toByteArray
          zipInputStream.closeEntry()
        }
    } finally {
      zipInputStream.close()
    }

    entries.toMap
  }

  private def findOpfPath(entries: Map[String, Array[Byte]]): String = {
    entries
      .get("META-INF/container.xml")
      .flatMap { containerBytes =>
        val containerXml = XML.loadString(decodeDocument(containerBytes))
        (containerXml \\ "rootfile").headOption
          .flatMap(node => node.attribute("full-path").map(_.text.trim))
      }
      .orElse(entries.keys.find(_.toLowerCase.endsWith(".opf")))
      .map(normalizePath)
      .getOrElse(throw new IllegalArgumentException("Could not locate EPUB package document"))
  }

  private def parseManifest(opfXml: scala.xml.Elem, opfPath: String): Seq[ManifestItem] = {
    (opfXml \\ "manifest" \\ "item").flatMap { node =>
      for {
        id <- node.attribute("id").map(_.text.trim)
        href <- node.attribute("href").map(_.text.trim)
        mediaType <- node.attribute("media-type").map(_.text.trim)
      } yield {
        val properties =
          node
            .attribute("properties")
            .map(_.text.trim.split("\\s+").filter(_.nonEmpty).toSet)
            .getOrElse(Set.empty[String])
        ManifestItem(id, resolveRelativePath(opfPath, href), mediaType, properties)
      }
    }
  }

  private def resolveSpine(
      opfXml: scala.xml.Elem,
      manifestItems: Seq[ManifestItem],
      manifestById: Map[String, ManifestItem],
      entryNames: Seq[String]): Seq[String] = {

    val spineItems = (opfXml \\ "spine" \\ "itemref").flatMap { node =>
      node.attribute("idref").map(_.text.trim).flatMap(manifestById.get)
    }

    val orderedSpine =
      spineItems.filter(item => htmlMediaTypes.contains(item.mediaType)).map(_.path)
    if (orderedSpine.nonEmpty) {
      orderedSpine
    } else {
      val manifestFallback =
        manifestItems.filter(item => htmlMediaTypes.contains(item.mediaType)).map(_.path)
      if (manifestFallback.nonEmpty) {
        manifestFallback
      } else {
        entryNames
          .filter(name =>
            name.toLowerCase.endsWith(".xhtml") || name.toLowerCase.endsWith(".html"))
          .sorted
      }
    }
  }

  private def decodeDocument(bytes: Array[Byte]): String = {
    val detectedCharset = detectEncoding(bytes)
    new String(bytes, detectedCharset)
  }

  private def detectEncoding(bytes: Array[Byte]): Charset = {
    val sample = new String(bytes.take(256), StandardCharsets.ISO_8859_1)
    val encodingRegex = """(?i)encoding=["']([^"']+)["']""".r
    val charsetName = encodingRegex.findFirstMatchIn(sample).map(_.group(1)).getOrElse("UTF-8")
    try Charset.forName(charsetName)
    catch {
      case _: Exception => StandardCharsets.UTF_8
    }
  }

  private def resolveArchiveEntry(
      basePath: String,
      targetPath: String,
      entries: Map[String, Array[Byte]]): Option[Array[Byte]] = {
    val resolvedPath = resolveRelativePath(basePath, targetPath)
    lookupEntry(entries, resolvedPath)
      .orElse(lookupEntry(entries, targetPath))
  }

  private def lookupEntry(
      entries: Map[String, Array[Byte]],
      path: String): Option[Array[Byte]] = {
    val normalized = normalizePath(stripFragmentAndQuery(path))
    entries.get(normalized).orElse {
      val decoded = decodePercentEncoding(normalized)
      if (decoded != normalized) entries.get(decoded) else None
    }
  }

  private def stripFragmentAndQuery(path: String): String = {
    path.takeWhile(ch => ch != '#' && ch != '?')
  }

  private def resolveRelativePath(basePath: String, href: String): String = {
    val cleanHref = stripFragmentAndQuery(href).replace('\\', '/')
    if (cleanHref.isEmpty) normalizePath(basePath)
    else if (cleanHref.startsWith("/")) normalizePath(cleanHref)
    else {
      val parent = parentPath(basePath)
      normalizePath(if (parent.nonEmpty) s"$parent/$cleanHref" else cleanHref)
    }
  }

  private def parentPath(path: String): String = {
    val normalized = normalizePath(path)
    val lastSlash = normalized.lastIndexOf('/')
    if (lastSlash >= 0) normalized.substring(0, lastSlash) else ""
  }

  private def normalizePath(path: String): String = {
    val pathWithoutScheme = path.stripPrefix("./").stripPrefix("/")
    val segments = mutable.ArrayBuffer.empty[String]
    pathWithoutScheme.replace('\\', '/').split("/").foreach {
      case "" | "." =>
      case ".." =>
        if (segments.nonEmpty) segments.remove(segments.length - 1)
      case part =>
        segments += part
    }
    segments.mkString("/")
  }

  private def decodePercentEncoding(path: String): String = {
    val builder = new StringBuilder
    var index = 0
    while (index < path.length) {
      if (path.charAt(index) == '%' && index + 2 < path.length) {
        val hex = path.substring(index + 1, index + 3)
        try {
          builder.append(Integer.parseInt(hex, 16).toChar)
          index += 3
        } catch {
          case _: NumberFormatException =>
            builder.append(path.charAt(index))
            index += 1
        }
      } else {
        builder.append(path.charAt(index))
        index += 1
      }
    }
    builder.toString()
  }

  private def decodeBase64Image(content: String): Option[Array[Byte]] = {
    try Some(Base64.getDecoder.decode(content))
    catch {
      case _: IllegalArgumentException => None
    }
  }

  private def parseDataUriMediaType(header: String): Option[String] = {
    val prefix = header.stripPrefix("data:")
    val mediaType = prefix.takeWhile(_ != ';')
    Option(mediaType).filter(_.nonEmpty)
  }

  private def inferMediaType(path: String): String = {
    val lowered = stripFragmentAndQuery(path).toLowerCase
    if (lowered.endsWith(".jpg") || lowered.endsWith(".jpeg")) "image/jpeg"
    else if (lowered.endsWith(".png")) "image/png"
    else if (lowered.endsWith(".gif")) "image/gif"
    else if (lowered.endsWith(".bmp")) "image/bmp"
    else if (lowered.endsWith(".svg")) "image/svg+xml"
    else "application/octet-stream"
  }

  private def parseInt(value: String): Option[Int] = {
    try Some(value.toInt)
    catch {
      case _: NumberFormatException => None
    }
  }
}
