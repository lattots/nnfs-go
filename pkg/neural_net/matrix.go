package neuralnet

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/lattots/gonum/pkg/matrix"
)

func randomMatrix(m, n int, initType InitializationType, seed int) (*matrix.Matrix, error) {
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

func xavierInitialization(rows, cols int, randGen *rand.Rand) (*matrix.Matrix, error) {
	limit := math.Sqrt(6.0 / float64(rows+cols))
	return randomMatrixWithRange(rows, cols, -limit, limit, randGen)
}

func heInitialization(rows, cols int, randGen *rand.Rand) (*matrix.Matrix, error) {
	stddev := math.Sqrt(2.0 / float64(rows))
	return randomMatrixWithStandardDeviation(rows, cols, stddev, randGen)
}

func randomMatrixWithRange(rows, cols int, min, max float64, randGen *rand.Rand) (*matrix.Matrix, error) {
	data := make([][]float64, rows)
	for i := range rows {
		data[i] = make([]float64, cols)
		for j := range cols {
			data[i][j] = min + (max-min)*randGen.Float64()
		}
	}
	return matrix.NewMatrix(data)
}

func randomMatrixWithStandardDeviation(rows, cols int, stddev float64, randGen *rand.Rand) (*matrix.Matrix, error) {
	data := make([][]float64, rows)
	for i := range rows {
		data[i] = make([]float64, cols)
		for j := range cols {
			data[i][j] = randGen.NormFloat64() * stddev // Use normal distribution
		}
	}
	return matrix.NewMatrix(data)
}

type InitializationType int

const (
	Xavier InitializationType = iota
	He
)

func addBias(m *matrix.Matrix, b []float64) error {
	if len(b) != m.N {
		return fmt.Errorf("matrix doesn't have the same number of columns as biases\n%d != %d", m.N, len(b))
	}
	for i := 0; i < m.M; i++ {
		for j := 0; j < m.N; j++ {
			m.Data[i][j] += b[j]
		}
	}
	return nil
}

func oneMatrix(m, n int) (*matrix.Matrix, error) {
	mat, err := matrix.NewZeroMatrix(m, n)
	if err != nil {
		return nil, fmt.Errorf("error creating zero matrix: %w", err)
	}
	for i := range m {
		for j := range n {
			mat.Data[i][j] = 1
		}
	}
	return mat, nil
}

// NormalizeMatrix scales all values in the matrix to range from 0-1 while preserving their relative sizes
func NormalizeMatrix(m *matrix.Matrix) (*matrix.Matrix, error) {
	var largestAbs float64 = 0
	for i := range m.Data {
		for j := range m.Data[i] {
			val := math.Abs(m.Data[i][j])
			if val > largestAbs {
				largestAbs = val
			}
		}
	}

	res, err := matrix.NewMatrix(m.Data)
	if err != nil {
		return nil, fmt.Errorf("error creating result matrix: %w", err)
	}
	res.Scale(1 / largestAbs)
	return res, nil
}
