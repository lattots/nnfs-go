package mathematic

import (
	"fmt"
	"math"

	"github.com/lattots/gonum/mat"
)

func Sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-x))))
}

func SigmoidPrime(x float32) float32 {
	return Sigmoid(x) * (1.0 - Sigmoid(x))
}

func ReLU(x float32) float32 {
	return float32(math.Max(0, float64(x)))
}

func ReLUPrime(x float32) float32 {
	if x > 0 {
		return 1
	}
	return 0
}

func Softmax(x []float32) []float32 {
	if len(x) == 0 {
		return nil
	}

	maxVal := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}

	result := make([]float32, len(x))
	sumExpX := 0.0

	for i := range x {
		expVal := math.Exp(float64(x[i] - maxVal))
		sumExpX += expVal

		result[i] = float32(expVal)
	}

	invSum := 1.0 / sumExpX

	for i := range result {
		result[i] = float32(float64(result[i]) * invSum)
	}

	return result
}

func SoftmaxMatrix(m *mat.Mat[float32]) *mat.Mat[float32] {
	data := make([]float32, len(m.Data))

	for i := range m.M {
		start := i * m.N
		end := (i + 1) * m.N
		softmaxedRow := Softmax(m.Data[start:end])
		copy(data[start:end], softmaxedRow)
	}

	return &mat.Mat[float32]{
		M:    m.M,
		N:    m.N,
		Data: data,
	}
}

func CrossEntropy(predicted, target *mat.Mat[float32]) (float32, error) {
	if predicted.M != target.M || predicted.N != target.N {
		return 0, fmt.Errorf("dimensions of predicted and target matrices must match")
	}

	loss := 0.0
	epsilon := 1e-15 // A small value to prevent log(0)

	for i := range predicted.Data {
		if target.Data[i] == 1 { // Only consider the true class for cross-entropy
			p := float64(predicted.Data[i])
			p = math.Max(epsilon, math.Min(1-epsilon, p))
			loss -= math.Log(p)
		}
	}
	return float32(loss / float64(predicted.M)), nil // Average over all samples
}

func CrossEntropyPrime(predicted, target *mat.Mat[float32]) (*mat.Mat[float32], error) {
	if predicted.M != target.M || predicted.N != target.N {
		return nil, fmt.Errorf("dimensions of predicted and target matrices must match")
	}

	derivative, err := mat.Zeros[float32](predicted.M, predicted.N)
	if err != nil {
		return nil, err
	}

	for i := range predicted.Data {
		if target.Data[i] == 1 {
			derivative.Data[i] = predicted.Data[i] - target.Data[i]
		}
	}

	return derivative, nil
}

func BinaryCrossEntropy(predicted, target *mat.Mat[float32]) (float32, error) {
	if predicted.M != target.M || predicted.N != target.N {
		return 0, fmt.Errorf("dimensions of predicted and target matrices must match")
	}

	loss := 0.0
	epsilon := 1e-15 // A small value to prevent log(0)

	for i := range predicted.Data {
		y := float64(target.Data[i])
		p := float64(predicted.Data[i])
		p = math.Max(epsilon, math.Min(1-epsilon, p))

		loss -= y*math.Log(p) + (1-y)*math.Log(1-p)
	}
	return float32(loss / float64(predicted.M)), nil // Average over all samples
}

func BinaryCrossEntropyPrime(predicted, target *mat.Mat[float32]) (*mat.Mat[float32], error) {
	if predicted.M != target.M || predicted.N != target.N {
		return nil, fmt.Errorf("dimensions of predicted and target matrices must match")
	}

	derivative, err := mat.Zeros[float32](predicted.M, predicted.N)
	if err != nil {
		return nil, err
	}

	for i := range predicted.Data {
		y := target.Data[i]
		p := predicted.Data[i]
		derivative.Data[i] = (p - y) / (p * (1 - p))
	}

	return derivative, nil
}

func Abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
