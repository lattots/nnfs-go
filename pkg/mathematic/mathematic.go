package mathematic

import (
	"fmt"
	"math"

	"github.com/lattots/gonum/pkg/matrix"
)

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidPrime(x float64) float64 {
	return Sigmoid(x) * (1.0 - Sigmoid(x))
}

func ReLU(x float64) float64 {
	return math.Max(0, x)
}

func ReLUPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func Softmax(x []float64) []float64 {
	expX := make([]float64, len(x))
	sumExpX := 0.0

	for i := range x {
		expX[i] = math.Exp(x[i])
		sumExpX += expX[i]
	}

	result := make([]float64, len(x))
	for i := range x {
		result[i] = expX[i] / sumExpX
	}

	return result
}

func SoftmaxMatrix(m *matrix.Matrix) (*matrix.Matrix, error) {
	result, err := matrix.NewZeroMatrix(m.M, m.N)
	if err != nil {
		return nil, err
	}

	for i := range m.Data {
		softmaxValues := Softmax(m.Data[i])
		for j := range softmaxValues {
			result.Data[i][j] = softmaxValues[j]
		}
	}

	return result, nil
}

func CrossEntropy(predicted *matrix.Matrix, target *matrix.Matrix) (float64, error) {
	if predicted.M != target.M || predicted.N != target.N {
		return 0, fmt.Errorf("dimensions of predicted and target matrices must match")
	}

	loss := 0.0
	epsilon := 1e-15 // A small value to prevent log(0)

	for i := 0; i < predicted.M; i++ {
		for j := 0; j < predicted.N; j++ {
			if target.Data[i][j] == 1 { // Only consider the true class for cross-entropy
				p := predicted.Data[i][j]
				p = math.Max(epsilon, math.Min(1-epsilon, p))
				loss -= math.Log(p)
			}
		}
	}
	return loss / float64(predicted.M), nil // Average over all samples
}

func CrossEntropyPrime(predicted *matrix.Matrix, target *matrix.Matrix) (*matrix.Matrix, error) {
	if predicted.M != target.M || predicted.N != target.N {
		return nil, fmt.Errorf("dimensions of predicted and target matrices must match")
	}

	derivative, err := matrix.NewZeroMatrix(predicted.M, predicted.N)
	if err != nil {
		return nil, err
	}

	for i := 0; i < predicted.M; i++ {
		for j := 0; j < predicted.N; j++ {
			if target.Data[i][j] == 1 {
				derivative.Data[i][j] = predicted.Data[i][j] - target.Data[i][j]
			}
		}
	}

	return derivative, nil
}

func BinaryCrossEntropy(predicted *matrix.Matrix, target *matrix.Matrix) (float64, error) {
	if predicted.M != target.M || predicted.N != target.N {
		return 0, fmt.Errorf("dimensions of predicted and target matrices must match")
	}

	loss := 0.0
	epsilon := 1e-15 // A small value to prevent log(0)

	for i := 0; i < predicted.M; i++ {
		for j := 0; j < predicted.N; j++ {
			y := target.Data[i][j]
			p := predicted.Data[i][j]
			p = math.Max(epsilon, math.Min(1-epsilon, p))

			loss -= y*math.Log(p) + (1-y)*math.Log(1-p)
		}
	}
	return loss / float64(predicted.M), nil // Average over all samples
}

func BinaryCrossEntropyPrime(predicted *matrix.Matrix, target *matrix.Matrix) (*matrix.Matrix, error) {
	if predicted.M != target.M || predicted.N != target.N {
		return nil, fmt.Errorf("dimensions of predicted and target matrices must match")
	}

	derivative, err := matrix.NewZeroMatrix(predicted.M, predicted.N)
	if err != nil {
		return nil, err
	}

	for i := 0; i < predicted.M; i++ {
		for j := 0; j < predicted.N; j++ {
			y := target.Data[i][j]
			p := predicted.Data[i][j]
			derivative.Data[i][j] = (p - y) / (p * (1 - p))
		}
	}

	return derivative, nil
}

func Abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
