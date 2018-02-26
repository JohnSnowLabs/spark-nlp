package com.johnsnowlabs.testutils

import scala.concurrent._
import scala.concurrent.ExecutionContext.Implicits.global

/**
  * Helper object used to throw output to the console in order to prevent the travis build to fail
  * because of console silence for 10 minutes.
  */
object TravisCIHelper {
  var promise: Promise[Unit] = _
  var future: Future[Unit] = _
  def startLogger = {
    promise = Promise[Unit]
    future = Future {
      for (i <- 0 until 10) {
        if (!promise.isCompleted) {
          println("Travis keepalive")
          Thread.sleep(2 * 1000 * 60)
        }
      }
    }
    Future firstCompletedOf Seq(promise.future, future)
  }

  def stopLogger = {
    promise.failure(new Exception)
  }
}
