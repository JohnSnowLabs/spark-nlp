
val scalaLangVersion = "2.11.11"
val sparkVersion = "2.1.1"
val scalaTestVersion = "3.0.0"

lazy val analyticsDependencies = Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided"
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
    name := "spark-nlp",
    version := "1.0.0",
    organization := "com.jsl.nlp",
    scalaVersion := scalaLangVersion,
    libraryDependencies ++=
      analyticsDependencies ++
      testDependencies ++
      utilDependencies
  )

/** Copies the assembled jar to the pyspark/lib dir **/
lazy val copyAssembledJar = taskKey[Unit]("Copy assembled jar to pyspark/lib")

copyAssembledJar := {
  val jarFilePath = (assemblyOutputPath in assembly).value
  val newJarFilePath = baseDirectory( _ / "pysparknlp" / "lib" /  "sparknlp.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}