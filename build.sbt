name := "spark-nlp"

organization := "jhonsnowlabs"

version := "1.0.0"

scalaVersion := "2.11.11"

val sparkVer = "2.1.1"

val scalaTestVersion = "3.0.0"

sparkVersion := sparkVer

lazy val analyticsDependencies = Seq(
  "org.apache.spark" %% "spark-core" % sparkVer % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVer % "provided"
)

lazy val testDependencies = Seq(
  "org.scalatest" %% "scalatest" % scalaTestVersion % "test"
)

lazy val utilDependencies = Seq(
  "com.typesafe" % "config" % "1.3.0"
)

parallelExecution in Test := false
logBuffered in Test := false

scalacOptions ++= Seq(
  "-feature",
  "-language:implicitConversions"
)

/** Enable for debugging */
testOptions in Test += Tests.Argument("-oF")

/** Disables tests in assembly */
test in assembly := {}

lazy val root = (project in file("."))
  .settings(
    libraryDependencies ++=
      analyticsDependencies ++
      testDependencies ++
      utilDependencies
  )

/** Copies the assembled jar to the pyspark/lib dir **/
lazy val copyAssembledJar = taskKey[Unit]("Copy assembled jar to pyspark/lib")

copyAssembledJar := {
  val jarFilePath = (assemblyOutputPath in assembly).value
  val newJarFilePath = baseDirectory( _ / "python" / "lib" /  "sparknlp.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}

// sbt-spark-package settings

spName := "jhonsnowlabs/spark-nlp"

sparkComponents ++= Seq("mllib")

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

spIncludeMaven := false

spAppendScalaVersion := false

assemblyOption in assembly := (assemblyOption in assembly).value.copy(
	includeScala = false
)

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

ivyScala := ivyScala.value map {
  _.copy(overrideScalaVersion = true)
}
