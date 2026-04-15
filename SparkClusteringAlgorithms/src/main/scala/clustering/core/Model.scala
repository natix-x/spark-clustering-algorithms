package clustering.core

import clustering.data.Point


trait Model extends Serializable {
    def predict(point: Point): Int
}
