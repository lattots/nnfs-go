package util

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/lattots/gonum/mat"
)

type Dataset struct {
	Inputs  *mat.Mat[float32]
	Targets *mat.Mat[float32]
}

func New(inputs, targets *mat.Mat[float32]) *Dataset {
	return &Dataset{Inputs: inputs, Targets: targets}
}

func LoadDataset(filename string, numClasses int) (*Dataset, error) {
	rawCSVData, err := ReadCSVToFloat32(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read csv %s: %w", filename, err)
	}

	labels, err := ExtractOutput(&rawCSVData, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to extract labels: %w", err)
	}

	inputMatrix, err := mat.New(rawCSVData)
	if err != nil {
		return nil, fmt.Errorf("failed to construct input matrix: %w", err)
	}

	inputNormalized, err := NormalizeMatrix(inputMatrix)
	if err != nil {
		return nil, fmt.Errorf("failed to normalize inputs: %w", err)
	}

	targetMatrix, err := ToOneHot(labels, numClasses)
	if err != nil {
		return nil, fmt.Errorf("failed to generate one-hot target matrix: %w", err)
	}

	return New(inputNormalized, targetMatrix), nil
}

type DataLoader struct {
	ds        *Dataset
	batchSize int
	shuffle   bool
	indices   []int
	current   int
	rng       *rand.Rand
}

func NewLoader(ds *Dataset, batchSize int, shuffle bool) *DataLoader {
	numSamples := ds.Inputs.M
	indices := make([]int, numSamples)
	for i := range indices {
		indices[i] = i
	}

	return &DataLoader{
		ds:        ds,
		batchSize: batchSize,
		shuffle:   shuffle,
		indices:   indices,
		rng:       rand.New(rand.NewSource(int64(time.Now().Nanosecond()))),
	}
}

func (dl *DataLoader) Reset() {
	dl.current = 0
	if !dl.shuffle {
		return
	}

	dl.rng.Shuffle(len(dl.indices), func(i, j int) {
		dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
	})
}

func (dl *DataLoader) HasNext() bool {
	return dl.current < len(dl.indices)
}

func (dl *DataLoader) NumBatches() int {
	return (len(dl.indices) + dl.batchSize - 1) / dl.batchSize
}

func (dl *DataLoader) NextBatch() (xBatch, yBatch *mat.Mat[float32]) {
	end := min(dl.current+dl.batchSize, len(dl.indices))

	batchIndices := dl.indices[dl.current:end]
	dl.current = end

	xBatch = mat.GetRows(dl.ds.Inputs, batchIndices)
	yBatch = mat.GetRows(dl.ds.Targets, batchIndices)

	return xBatch, yBatch
}
