package neuralnet

import (
	"testing"
)

func TestNewRandomLayer(t *testing.T) {
	const (
		inputSize      = 12
		neuronCount    = 8
		initType       = Xavier
		activationType = ReLU
	)
	conf := &LayerConfig{
		InputSize:      inputSize,
		NeuronCount:    neuronCount,
		ActivationType: activationType,
		InitType:       initType,
	}

	l := newRandomLayer(conf)
	if l == nil {
		t.Fatal("failed to create random layer")
	}
	if l.weights.M != inputSize {
		t.Errorf("invalid number of weights per neuron: %d != %d\nweights matrix:\n%s", l.weights.M, inputSize, l.weights.String())
	}
	if l.weights.N != neuronCount {
		t.Errorf("invalid number of neurons in weights: %d != %d\nweights matrix:\n%s", l.weights.N, neuronCount, l.weights.String())
	}
	if l.biases.N != neuronCount {
		t.Errorf("invalid number of biases: %d != %d\nbiases matrix:\n%s", l.biases.N, neuronCount, l.biases.String())
	}
}

func TestSigmoidPass(t *testing.T) {
	const (
		inputSize      = 12
		neuronCount    = 8
		initType       = Xavier
		activationType = Sigmoid
	)
	conf := &LayerConfig{
		InputSize:      inputSize,
		NeuronCount:    neuronCount,
		ActivationType: activationType,
		InitType:       initType,
	}

	l := newRandomLayer(conf)
	if l == nil {
		t.Fatal("failed to create random layer")
	}

	input, err := randomMatrix(1, inputSize, initType)
	if err != nil {
		t.Error(err)
	}
	r, err := l.forward(input)
	if r == nil || err != nil {
		t.Error("failed to do sigmoid forward pass:", err)
	}
}
