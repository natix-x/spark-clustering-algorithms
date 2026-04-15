name := "spark-clustering"

version := "0.1"

scalaVersion := "2.12.21"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.5.8",
  "org.apache.spark" %% "spark-mllib" % "3.5.8"
)