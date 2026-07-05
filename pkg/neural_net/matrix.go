package neuralnet

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/lattots/gonum/mat"
)

func randomMatrix(m, n int, initType InitializationType, seed int) (*mat.Mat[float32], error) {
	randSrc := rand.NewSource(int64(seed))
	randGen := rand.New(randSrc)
	switch initType {
	case Xavier:
		return xavierInitialization(m, n, randGen)
	case He:
		return heInitialization(m, n, randGen)
	default: // Default to Xavier if no type is specified or invalid
		return xavierInitialization(m, n, randGen)
	}
}

func xavierInitialization(rows, cols int, randGen *rand.Rand) (*mat.Mat[float32], error) {
	limit := float32(math.Sqrt(6.0 / float64(rows+cols)))
	return randomMatrixWithRange(rows, cols, -limit, limit, randGen)
}

func heInitialization(rows, cols int, randGen *rand.Rand) (*mat.Mat[float32], error) {
	stddev := float32(math.Sqrt(2.0 / float64(rows)))
	return randomMatrixWithStandardDeviation(rows, cols, stddev, randGen)
}

func randomMatrixWithRange(rows, cols int, min, max float32, randGen *rand.Rand) (*mat.Mat[float32], error) {
	data := make([][]float32, rows)
	for i := range rows {
		data[i] = make([]float32, cols)
		for j := range cols {
			data[i][j] = min + (max-min)*randGen.Float32()
		}
	}
	return mat.New(data)
}

func randomMatrixWithStandardDeviation(rows, cols int, stddev float32, randGen *rand.Rand) (*mat.Mat[float32], error) {
	data := make([][]float32, rows)
	for i := range rows {
		data[i] = make([]float32, cols)
		for j := range cols {
			data[i][j] = float32(randGen.NormFloat64()) * stddev // Use normal distribution
		}
	}
	return mat.New(data)
}

type InitializationType int

const (
	Xavier InitializationType = iota
	He
)

func addBias(m, b *mat.Mat[float32]) error {
	if !b.IsVector() {
		return fmt.Errorf("bias must be a vector")
	}

	m = mat.AddRowVector(m, b)

	return nil
}

// NormalizeMatrix scales all values in the matrix to range from 0-1 while preserving their relative sizes
func NormalizeMatrix(m *mat.Mat[float32]) (*mat.Mat[float32], error) {
	var largestAbs float32 = 0
	for i := range m.Data {
		val := float32(math.Abs(float64(m.Data[i])))
		if val > largestAbs {
			largestAbs = val
		}
	}

	res := mat.Scale(m, 1/largestAbs)

	return res, nil
}
