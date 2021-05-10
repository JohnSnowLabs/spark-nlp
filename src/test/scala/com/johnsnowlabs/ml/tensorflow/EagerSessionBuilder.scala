package com.johnsnowlabs.ml.tensorflow

import org.scalatest.{BeforeAndAfterAll, Suite}
import org.tensorflow.EagerSession
import org.tensorflow.op.Scope

trait EagerSessionBuilder extends BeforeAndAfterAll { this: Suite =>

  private val session: EagerSession = EagerSession.create
  val scope = new Scope(session)

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  override def afterAll(): Unit = {
    try super.afterAll()
    finally session.close()
  }

}
