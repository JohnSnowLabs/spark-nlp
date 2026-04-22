package com.johnsnowlabs.util

import com.sun.net.httpserver.{HttpExchange, HttpHandler, HttpServer}

import java.io.OutputStream
import java.net.InetSocketAddress

private[johnsnowlabs] object LocalHttpTestServer {

  final case class Response(
      statusCode: Int,
      body: Array[Byte],
      contentType: String = "text/plain",
      headers: Map[String, String] = Map.empty)

  final class RunningServer private[LocalHttpTestServer] (private val server: HttpServer)
      extends AutoCloseable {

    def url(path: String): String = {
      val normalizedPath = if (path.startsWith("/")) path else s"/$path"
      s"http://127.0.0.1:${server.getAddress.getPort}$normalizedPath"
    }

    override def close(): Unit = server.stop(0)
  }

  def start(routes: Map[String, Response]): RunningServer = {
    val server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0)

    routes.foreach { case (rawPath, response) =>
      val path = if (rawPath.startsWith("/")) rawPath else s"/$rawPath"
      server.createContext(
        path,
        (exchange: HttpExchange) => {
          response.headers.foreach { case (key, value) =>
            exchange.getResponseHeaders.add(key, value)
          }
          exchange.getResponseHeaders.add("Content-Type", response.contentType)
          exchange.sendResponseHeaders(response.statusCode, response.body.length.toLong)

          val outputStream: OutputStream = exchange.getResponseBody
          try {
            outputStream.write(response.body)
          } finally {
            outputStream.close()
            exchange.close()
          }
        })
    }

    server.start()
    new RunningServer(server)
  }
}
