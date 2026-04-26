package clustering.data

import java.util.Arrays

/**
 * Zoptymalizowana klasa reprezentująca punkt w n-wymiarowej przestrzeni.
 * Wykorzystuje natywne tablice (Array) dla maksymalnej wydajności w Sparku.
 */
class Point(val values: Array[Double]) extends Serializable {

  def dimension: Int = values.length

  // ==========================================
  // OPERATORY ALGEBRAICZNE (Wydajność Imperatywna)
  // Używamy bezalokacyjnych pętli 'while' zamiast 'zip().map()'
  // ==========================================

  /** Dodawanie dwóch wektorów (Niezbędne do reduceByKey w K-Means) */
  def +(other: Point): Point = {
    require(this.dimension == other.dimension, "Punkty muszą mieć ten sam wymiar")
    val newValues = new Array[Double](dimension)
    var i = 0
    while (i < dimension) {
      newValues(i) = this.values(i) + other.values(i)
      i += 1
    }
    new Point(newValues)
  }

  /** Odejmowanie dwóch wektorów (Przydatne w innych metrykach odległości) */
  def -(other: Point): Point = {
    require(this.dimension == other.dimension, "Punkty muszą mieć ten sam wymiar")
    val newValues = new Array[Double](dimension)
    var i = 0
    while (i < dimension) {
      newValues(i) = this.values(i) - other.values(i)
      i += 1
    }
    new Point(newValues)
  }

  /** Dzielenie przez skalar (Niezbędne do wyliczania średniej / nowych centroidów) */
  def /(scalar: Long): Point = this / scalar.toDouble

  def /(scalar: Double): Point = {
    val newValues = new Array[Double](dimension)
    var i = 0
    while (i < dimension) {
      newValues(i) = this.values(i) / scalar
      i += 1
    }
    new Point(newValues)
  }

  /** Mnożenie przez skalar (Przydatne m.in. w GMM przy responsibilities) */
  def *(scalar: Double): Point = {
    val newValues = new Array[Double](dimension)
    var i = 0
    while (i < dimension) {
      newValues(i) = this.values(i) * scalar
      i += 1
    }
    new Point(newValues)
  }

  // ==========================================
  // METODY SYSTEMOWE (Krytyczne przy użyciu Array)
  // ==========================================

  // Ponieważ Array w JVM porównuje się przez referencję (adres w pamięci),
  // musimy nadpisać equals i hashCode, aby Spark poprawnie grupował punkty.
  override def equals(obj: Any): Boolean = obj match {
    case p: Point => Arrays.equals(this.values, p.values)
    case _ => false
  }

  override def hashCode(): Int = Arrays.hashCode(values)

  override def toString: String = s"Point(${values.mkString(", ")})"
}

// ==========================================
// COMPANION OBJECT
// Pozwala na tworzenie punktów bez słowa 'new', np. Point(Array(1.0, 2.0))
// ==========================================
object Point {
  def apply(values: Array[Double]): Point = new Point(values)

  // Przeciążenie dla kompatybilności wstecznej, jeśli masz gdzieś w kodzie Vector
  def apply(values: Vector[Double]): Point = new Point(values.toArray)
}