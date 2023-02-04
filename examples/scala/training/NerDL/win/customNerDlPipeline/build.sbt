import play.sbt.PlaySettings
import sbt.Keys._

lazy val GatlingTest = config("gatling") extend Test

scalaVersion := "2.11.12"

// Play Framework Dependencies
libraryDependencies += guice
libraryDependencies += "io.lemonlabs" %% "scala-uri" % "1.4.8"
libraryDependencies += "net.logstash.logback" % "logstash-logback-encoder" % "5.2" exclude("com.fasterxml.jackson.core", "jackson-annotations")
libraryDependencies += "com.netaporter" %% "scala-uri" % "0.4.14"
libraryDependencies += "net.codingwell" %% "scala-guice" % "4.1.0"
libraryDependencies += "org.joda" % "joda-convert" % "1.8.1"

// test dependencies
libraryDependencies += "org.scalatestplus.play" %% "scalatestplus-play" % "2.0.0" % Test
libraryDependencies += "io.gatling.highcharts" % "gatling-charts-highcharts" % "2.2.5" % Test
libraryDependencies += "io.gatling" % "gatling-test-framework" % "2.2.5" % Test
libraryDependencies += "org.mockito" % "mockito-all" % "1.9.5" % "test"

// Conflicting Dependencies
libraryDependencies += "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.9.9"
libraryDependencies += "io.netty" % "netty-transport" % "4.1.34.Final"

// Preprocessing Dependencies
libraryDependencies += "org.jsoup" % "jsoup" % "1.12.1"
//libraryDependencies += "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.6.7.1"
//libraryDependencies += "com.fasterxml.jackson" %% "jackson-databind" % "2.6.7.1"
//libraryDependencies += "com.fasterxml.jackson.core" % "jackson-core" % "2.9.9"
libraryDependencies += "com.crealytics" %% "spark-excel" % "0.12.0"
libraryDependencies += "info.folone" %% "poi-scala" % "0.19"
libraryDependencies += "com.jsuereth" %% "scala-arm" % "1.4"



// ML Dependencies
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "2.2.1"
//libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "2.2.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.3"
libraryDependencies += "net.sourceforge.f2j" % "arpack_combined_all" % "0.1"
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.3"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.3"

//dependencyOverrides += "com.google.guava" % "guava" % "15.0"
//dependencyOverrides += "com.google.guava" % "guava" % "21.0"
dependencyOverrides += "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "2.7.2"
dependencyOverrides += "org.apache.hadoop" % "hadoop-common" % "2.7.2"
dependencyOverrides += "commons-io" % "commons-io" % "2.4"

//excludeDependencies ++= Seq(
//  ExclusionRule(organization = "com.fasterxml.jackson")
//)

// The Play project itself
lazy val root = (project in file("."))
  .enablePlugins(Common, PlayService, PlayLayoutPlugin, GatlingPlugin)
  .configs(GatlingTest)
  .settings(inConfig(GatlingTest)(Defaults.testSettings): _*)
  .settings(
    name := """ScalaUtilsForML""",
    scalaSource in GatlingTest := baseDirectory.value / "/gatling/simulation"
  )

// Documentation for this project:
//    sbt "project docs" "~ paradox"
//    open docs/target/paradox/site/index.html
lazy val docs = (project in file("docs")).enablePlugins(ParadoxPlugin).
  settings(
    paradoxProperties += ("download_url" -> "https://example.lightbend.com/v1/download/play-rest-api")
  )
