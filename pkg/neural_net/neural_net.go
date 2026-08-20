package neuralnet

import (
	"fmt"
	"time"

	"github.com/lattots/gonum/mat"

	"github.com/lattots/nnfs_go/pkg/mathematic"
	"github.com/lattots/nnfs_go/pkg/util"
)

type Net struct {
	config Config
	layers []*layer
}

func NewNet(config Config) (*Net, error) {
	net := &Net{config: config}
	seed := time.Now().Nanosecond()
	if config.Seed != 0 {
		seed = config.Seed
	}
	err := net.initRandom(seed)
	if err != nil {
		return nil, err
	}
	return net, nil
}

type Config struct {
	NumEpochs    int
	LearningRate float32
	LossFunction LossFunctionType

	Seed int

	LayerConfigs []*LayerConfig
}

func (n *Net) initRandom(seed int) error {
	n.layers = make([]*layer, len(n.config.LayerConfigs))

	for i := 0; i < len(n.config.LayerConfigs); i++ {
		if n.config.LayerConfigs[i].InputSize == 0 {
			n.config.LayerConfigs[i].InputSize = n.config.LayerConfigs[i-1].NeuronCount
		}
		n.layers[i] = newRandomLayer(n.config.LayerConfigs[i], seed)
	}

	return nil
}

func (n *Net) Train(loader *util.DataLoader) error {
	for epoch := range n.config.NumEpochs {
		loader.Reset()

		totalLoss := float32(0.0)

		for loader.HasNext() {
			xBatch, yBatch := loader.NextBatch()

			zs, activations, err := n.forwardPassAndGetOutputs(xBatch)
			if err != nil {
				return fmt.Errorf("failed to forward pass: %w", err)
			}

			finalOutput := activations[len(activations)-1]
			softmaxOutput := mathematic.SoftmaxMatrix(finalOutput)

			loss, err := getLossFunction(n.config.LossFunction)(softmaxOutput, yBatch)
			if err != nil {
				return fmt.Errorf("error calculating loss: %w", err)
			}
			totalLoss += loss

			if err = n.backwardPass(yBatch, zs, activations, softmaxOutput); err != nil {
				return fmt.Errorf("backward pass failed: %w", err)
			}
		}

		fmt.Printf("Epoch: %d, Avg Loss: %.4f\n", epoch, totalLoss/float32(loader.NumBatches()))
	}
	return nil
}

func (n *Net) forwardPassAndGetOutputs(x *mat.Mat[float32]) (zs, activations []*mat.Mat[float32], err error) {
	// Pre-pend the input matrix to a list of activations for easier backprop later
	activations = []*mat.Mat[float32]{x}
	currentInput := x

	for i, layer := range n.layers {
		// Calculate pre-activation z = (input * weights) + biases
		z, err := layer.forwardPreActivation(currentInput)
		if err != nil {
			return nil, nil, fmt.Errorf("error in pre-activation for layer %d: %w", i, err)
		}
		zs = append(zs, z)

		// Calculate activation a = g(z)
		a, err := layer.forwardActivation(z)
		if err != nil {
			return nil, nil, fmt.Errorf("error in activation for layer %d: %w", i, err)
		}
		activations = append(activations, a)
		currentInput = a // The output of this layer is the input to the next
	}

	return zs, activations, nil
}

func (n *Net) backwardPass(y *mat.Mat[float32], zs, activations []*mat.Mat[float32], softmaxOutput *mat.Mat[float32]) error {
	// For Softmax + CCE, the initial delta is simply (prediction - actual)
	finalDelta := mat.Subtract(softmaxOutput, y)

	deltas := make([]*mat.Mat[float32], len(n.layers))
	deltas[len(deltas)-1] = finalDelta

	// Calculate deltas for hidden layers (from right to left)
	for i := len(n.layers) - 2; i >= 0; i-- {
		errorPropagated, err := mat.Dot(deltas[i+1], mat.T(n.layers[i+1].weights))
		if err != nil {
			return err
		}

		activationDerivative := getActivationDerivative(n.layers[i].activationType)
		slope := mat.Map(zs[i], activationDerivative)

		deltas[i], err = mat.Mul(errorPropagated, slope)
		if err != nil {
			return err
		}
	}

	batchSize := float32(activations[0].M) // activations[0] is the input x
	for i := len(n.layers) - 1; i >= 0; i-- {
		// The input to layer 'i' is the activation from the previous layer, 'i'.
		// activations = [x, a_layer0, a_layer1, ...]
		prevLayerActivations := activations[i]

		weightGradient, err := mat.Dot(mat.T(prevLayerActivations), deltas[i])
		if err != nil {
			return err
		}
		weightGradient = mat.Scale(weightGradient, n.config.LearningRate/batchSize)
		n.layers[i].weights = mat.Subtract(n.layers[i].weights, weightGradient)

		biasGradient := mat.SumRows(deltas[i])
		biasGradient = mat.Scale(biasGradient, n.config.LearningRate/batchSize)
		n.layers[i].biases = mat.Subtract(n.layers[i].biases, biasGradient)
	}

	return nil
}

func (n *Net) Predict(x *mat.Mat[float32]) (*mat.Mat[float32], error) {
	res, err := n.forwardPass(x)
	if err != nil {
		return nil, fmt.Errorf("error forward passing: %w", err)
	}
	res = mathematic.SoftmaxMatrix(res)

	return res, nil
}

// Evaluate returns an accuracy score by running batch predictions through the provided DataLoader.
func (n *Net) Evaluate(loader *util.DataLoader) (float32, error) {
	loader.Reset()

	var correct int
	var totalSamples int

	for loader.HasNext() {
		xBatch, yBatch := loader.NextBatch()

		outputBatch, err := n.Predict(xBatch)
		if err != nil {
			return 0, fmt.Errorf("error making prediction: %w", err)
		}

		batchSize := xBatch.M
		numClasses := yBatch.N

		for i := 0; i < batchSize; i++ {
			// Find argmax for predicted probabilities
			predIdx := 0
			maxPredVal := outputBatch.Data[i*numClasses]

			// Find argmax for target one-hot vector
			targetIdx := 0
			maxTargetVal := yBatch.Data[i*numClasses]

			for j := 1; j < numClasses; j++ {
				pVal := outputBatch.Data[i*numClasses+j]
				if pVal > maxPredVal {
					maxPredVal = pVal
					predIdx = j
				}

				tVal := yBatch.Data[i*numClasses+j]
				if tVal > maxTargetVal {
					maxTargetVal = tVal
					targetIdx = j
				}
			}

			if predIdx == targetIdx {
				correct++
			}
		}

		totalSamples += batchSize
	}

	if totalSamples == 0 {
		return 0, fmt.Errorf("loader provided zero samples for evaluation")
	}

	return float32(correct) / float32(totalSamples), nil
}

func (n *Net) forwardPass(x *mat.Mat[float32]) (*mat.Mat[float32], error) {
	res, err := n.layers[0].forward(x)
	if err != nil {
		return nil, err
	}
	for i := 1; i < len(n.layers); i++ { // loop through all
		res, err = n.layers[i].forward(res)
		if err != nil {
			return nil, fmt.Errorf("error forward passing through layer: %w", err)
		}
	}

	return res, err
}
