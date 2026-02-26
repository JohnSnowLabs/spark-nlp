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
package com.johnsnowlabs.reader.util

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.Build
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.rendering.{ImageType, PDFRenderer}

import java.awt.image.BufferedImage
import java.io._
import java.net.{HttpURLConnection, URL}
import java.util.Base64
import javax.imageio.ImageIO
import scala.annotation.tailrec
import scala.util.{Try, Using}

object ImageParser {

  /** Decodes a base64-encoded string into a BufferedImage.
    *
    * @param base64Str
    *   Base64 encoded string (without "data:image/png;base64," prefix).
    * @return
    *   Option[BufferedImage] if the bytes can be decoded by ImageIO.
    */
  def decodeBase64(base64Str: String): Option[BufferedImage] = {
    val cleaned = base64Str.replaceAll("\\s", "")
    val bytes = Base64.getDecoder.decode(cleaned)
    if (bytes == null || bytes.isEmpty) return None
    Using.resource(new ByteArrayInputStream(bytes)) { in =>
      Try(ImageIO.read(in)).toOption // catch exception → None
    }
  }

  /** Fetches an image from a remote URL and decodes it into a BufferedImage.
    *
    * Some CDNs (incl. Wikimedia) return 403 to non-browser clients without a descriptive
    * User-Agent. We set one and handle redirects & error bodies.
    *
    * @param urlStr
    *   Image URL (e.g. https://.../image.png)
    * @param connectTimeoutMs
    *   Connect timeout in milliseconds
    * @param readTimeoutMs
    *   Read timeout in milliseconds
    * @param headers
    *   Additional request headers
    * @param maxRedirects
    *   Max number of redirects to follow
    * @return
    *   Option[BufferedImage] if the stream can be decoded by ImageIO
    */
  def fetchFromUrl(
      urlStr: String,
      connectTimeoutMs: Int = 1000,
      readTimeoutMs: Int = 1000,
      headers: Map[String, String] = Map.empty,
      maxRedirects: Int = 5): Option[BufferedImage] = {

    @tailrec
    def fetch(url: URL, redirectsLeft: Int): Option[BufferedImage] = {
      val userAgent = buildDefaultUserAgent()
      val conn = open(url, connectTimeoutMs, readTimeoutMs, userAgent, headers)
      val code = conn.getResponseCode
      if (isRedirect(code)) {
        if (redirectsLeft <= 0) {
          conn.disconnect()
          throw new IOException(s"Too many redirects when fetching $url")
        }
        val location = Option(conn.getHeaderField("Location")).getOrElse {
          conn.disconnect(); throw new IOException(s"Redirect without Location from $url")
        }
        val nextUrl = new URL(url, location)
        conn.disconnect()
        fetch(nextUrl, redirectsLeft - 1)
      } else if (code >= 200 && code < 300) {
        Using.resource(new BufferedInputStream(conn.getInputStream)) { in =>
          Option(ImageIO.read(in)) // may return None when format unsupported
        }
      } else {
        val snippet = readErrorSnippet(conn.getErrorStream)
        val message =
          if (snippet.nonEmpty) s"HTTP $code for $url: $snippet" else s"HTTP $code for $url"
        conn.disconnect()
        throw new IOException(message)
      }
    }

    if (ResourceHelper.isHTTPProtocol(urlStr)) {
      fetch(new URL(urlStr), maxRedirects)
    } else None
  }

  private def buildDefaultUserAgent(): String = {
    val libVersion = Build.version
    val javaV = System.getProperty("java.version", "?")
    val scalaV = util.Properties.versionNumberString
    s"JohnSnowLabs-SparkNLP/$libVersion (Java/$javaV; Scala/$scalaV; +https://sparknlp.org)"
  }

  private def isRedirect(code: Int): Boolean =
    code match {
      case 301 | 302 | 303 | 307 | 308 => true
      case _ => false
    }

  private def open(
      url: URL,
      connectTimeoutMs: Int,
      readTimeoutMs: Int,
      userAgent: String,
      headers: Map[String, String]): HttpURLConnection = {
    val conn = url.openConnection().asInstanceOf[HttpURLConnection]
    conn.setInstanceFollowRedirects(false) // handle ourselves to preserve method/headers
    conn.setRequestMethod("GET")
    conn.setConnectTimeout(connectTimeoutMs)
    conn.setReadTimeout(readTimeoutMs)
    conn.setRequestProperty("User-Agent", userAgent) // avoid 403 from strict CDNs
    conn.setRequestProperty("Accept", "image/*,*/*;q=0.8")
    headers.foreach { case (k, v) => conn.setRequestProperty(k, v) }
    conn
  }

  private def readErrorSnippet(err: InputStream): String = {
    if (err == null) return ""
    Using.resource(err) { resource =>
      val out = new ByteArrayOutputStream()
      val buffer = new Array[Byte](512)
      var errorData = resource.read(buffer)
      while (errorData != -1 && out.size() < 4096) {
        out.write(buffer, 0, errorData); errorData = resource.read(buffer)
      }
      new String(out.toByteArray, java.nio.charset.StandardCharsets.UTF_8).linesIterator
        .take(3)
        .mkString(" ") // keep message short
    }
  }

  /** Decodes raw image bytes into a BufferedImage.
    *
    * @param bytes
    *   Raw image data (e.g. extracted from Word DOC/DOCX via Apache POI).
    * @return
    *   Option[BufferedImage] if the bytes can be decoded by ImageIO.
    */
  def bytesToBufferedImage(bytes: Array[Byte]): Option[BufferedImage] = {
    if (bytes == null || bytes.isEmpty) return None
    Using.resource(new ByteArrayInputStream(bytes)) { in =>
      Try(ImageIO.read(in)).toOption // returns None if format unsupported
    }
  }

  /** Renders each page of a PDF document into a BufferedImage.
    *
    * @param pdfContent
    *   Raw PDF bytes.
    * @return
    *   Map of page index (0-based) to Option[BufferedImage] for each page that could be rendered.
    *   If a page cannot be rendered, its value will be None.
    */
  def renderPdfFile(pdfContent: Array[Byte]): Map[Int, Option[BufferedImage]] = {
    val document = PDDocument.load(pdfContent)

    try {
      val renderer = new PDFRenderer(document)
      val ocrDpiQuality = 150

      (0 until document.getNumberOfPages).map { pageIndex =>
        val imageBuffer = renderer.renderImageWithDPI(pageIndex, ocrDpiQuality, ImageType.RGB)
        pageIndex -> Some(imageBuffer)
      }.toMap
    } finally {
      document.close()
    }
  }

}
