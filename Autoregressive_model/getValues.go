package main

import "C"

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"unsafe"

	//"sync"
	"time"
	//"unsafe"
	//"gonum.org/v1/gonum/mat"
)

type policyParams struct {
	//***********************************************************************************************************************************
	params []float64
	//***********************************************************************************************************************************
}

type gaussianParams struct {
	//***********************************************************************************************************************************
	s1max float64
	s1min float64
	s2max float64
	s2min float64
	//***********************************************************************************************************************************
}

type transitionParams struct {
	//***********************************************************************************************************************************
	theta1 []float64
	theta2 []float64
	sd1    float64
	sd2    float64
	//***********************************************************************************************************************************
}

func policy(state []float64, pParams policyParams, generator *rand.Rand) (int, float64) {
	lp := pParams.params[0] + state[0]*pParams.params[1] + state[1]*pParams.params[2] + state[0]*state[1]*pParams.params[3]
	if lp > 100. {
		lp = 100.
	}
	prob := math.Exp(lp) / (1 + math.Exp(lp))
	temp := generator.Float64()
	if temp < prob {
		return 1, prob
	}
	return 0, 1 - prob
}

func getProb(state []float64, action int, pParams policyParams) float64 {
	lp := pParams.params[0] + state[0]*pParams.params[1] + state[1]*pParams.params[2] + state[0]*state[1]*pParams.params[3]
	if lp > 100. {
		lp = 100.
	}
	prob := math.Exp(lp) / (1 + math.Exp(lp))
	if action == 1 {
		return prob
	}
	return 1 - prob
}

func randomNormal(mean, sd float64, generator *rand.Rand) float64 {
	return mean + sd*generator.NormFloat64()
}

func transition(state []float64, action int, tParams transitionParams, generator *rand.Rand) []float64 {
	//***********************************************************************************************************************************
	//fmt.Println(state)
	theta1 := tParams.theta1
	theta2 := tParams.theta2
	sd1 := tParams.sd1
	sd2 := tParams.sd2
	newState := make([]float64, 2)
	newState[0] = theta1[0] + state[0]*theta1[1] + state[1]*theta1[2] + state[0]*state[1]*theta1[3] + float64(action)*theta1[4] + float64(action)*state[0]*theta1[5] + float64(action)*state[1]*theta1[6] + randomNormal(0.0, sd1, generator)
	newState[1] = theta2[0] + state[0]*theta2[1] + state[1]*theta2[2] + state[0]*state[1]*theta2[3] + float64(action)*theta2[4] + float64(action)*state[0]*theta2[5] + float64(action)*state[1]*theta2[6] + randomNormal(0.0, sd2, generator)

	if newState[0] > 999.0 {
		newState[0] = 999.0
	} else if newState[0] < -999.0 {
		newState[0] = -999.0
	}
	if newState[1] > 999.0 {
		newState[1] = 999.0
	} else if newState[1] < -999.0 {
		newState[1] = -999.0
	}
	//***********************************************************************************************************************************
	return newState
}

func getReward(nextState []float64, action int) float64 {
	return 2.0*nextState[0] + nextState[1] - 0.25*(2.0*float64(action)-1)
}

func valueChain(state []float64, discount float64, chainLength int, pParams policyParams, tParams transitionParams, generator *rand.Rand, valueLocation *float64) {
	var action int
	total := 0.0
	for i := 0; i < chainLength; i++ {
		action, _ = policy(state, pParams, generator)
		state = transition(state, action, tParams, generator)
		total += math.Pow(discount, float64(i)) * getReward(state, action)
	}
	*valueLocation = total
	return
}

func averageFloat64(vec []float64) float64 {
	n := len(vec)
	tot := 0.0
	for i := 0; i < n; i++ {
		tot += vec[i]
	}
	return tot / float64(n)
}

func estValue(state []float64, discount float64, chainLength, numChain int, pParams policyParams, tParams transitionParams, valueLocation *float64) {
	valueVec := make([]float64, numChain)
	numRoutines := 20
	var ch = make(chan int, numChain) // This number 50 can be anything as long as it's larger than xthreads
	var wg sync.WaitGroup
	wg.Add(numRoutines)
	for i := 0; i < numRoutines; i++ {
		go func() {
			for {
				a, ok := <-ch
				if !ok { // if there is nothing to do and the channel has been closed then end the goroutine
					wg.Done()
					return
				}
				source := rand.NewSource(time.Now().UnixNano())
				generator := rand.New(source)
				valueChain(state, discount, chainLength, pParams, tParams, generator, &valueVec[a])
			}
		}()
	}

	for i := 0; i < numChain; i++ {
		ch <- i // add i to the queue
	}

	close(ch) // This tells the goroutines there's nothing else to do
	wg.Wait()

	*valueLocation = averageFloat64(valueVec)

	return
}

func setValues(states [][]float64, discount float64, chainLength, numChain int, pParams policyParams, tParams transitionParams) []float64 {
	n := len(states)
	valueVec := make([]float64, len(states))
	numRoutines := 50
	var ch = make(chan int, n) // This number 50 can be anything as long as it's larger than xthreads
	var wg sync.WaitGroup
	wg.Add(numRoutines)
	for i := 0; i < numRoutines; i++ {
		go func() {
			for {
				a, ok := <-ch
				if !ok { // if there is nothing to do and the channel has been closed then end the goroutine
					wg.Done()
					return
				}
				estValue(states[a], discount, chainLength, numChain, pParams, tParams, &valueVec[a])
			}
		}()
	}

	for i := 0; i < n; i++ {
		ch <- i // add i to the queue
	}

	close(ch) // This tells the goroutines there's nothing else to do
	wg.Wait() // Wait for the threads to finish

	return valueVec
}

func createTransPars(theta1, theta2 []float64, sd1, sd2 float64) transitionParams {
	var tParams transitionParams
	if len(theta1) < 7 {
		theta11 := make([]float64, 7)
		theta22 := make([]float64, 7)
		for i := 0; i < 6; i++ {
			if i < 4 {
				theta11[i] = theta1[i]
				theta22[i] = theta2[i]
			} else {
				theta11[i+1] = theta1[i]
				theta22[i+1] = theta2[i]
			}
		}
		tParams = transitionParams{theta11, theta22, sd1, sd2}
	} else {
		tParams = transitionParams{theta1, theta2, sd1, sd2}
	}
	return tParams
}

func createPolicyPars(polPars []float64) policyParams {
	pParams := policyParams{polPars}
	return pParams
}

// PySetValues does Monte Carlo simulation to find the values of the provided states under the given environment parameters
//export PySetValues
func PySetValues(statesVec, polPars, theta1, theta2 []float64, cSd1, cSd2 C.double, cChainLength, cNumChain C.int, cDiscount C.double, cMaxProcs C.int) uintptr {

	runtime.GOMAXPROCS(int(cMaxProcs))

	numRows := len(statesVec) / 2
	stateMat := make([][]float64, numRows)

	for i := 0; i < numRows; i++ {
		stateMat[i] = []float64{statesVec[2*i], statesVec[2*i+1]}
	}

	chainLength := int(cChainLength)
	numChain := int(cNumChain)
	discount := float64(cDiscount)
	sd1 := float64(cSd1)
	sd2 := float64(cSd2)

	tParams := createTransPars(theta1, theta2, sd1, sd2)
	pParams := createPolicyPars(polPars)

	values := setValues(stateMat, discount, chainLength, numChain, pParams, tParams)

	return uintptr(unsafe.Pointer(&values[0]))

}

func setGaussianParams(states [][]float64, gPars *gaussianParams) {
	s1max := -10.
	s1min := 10.
	s2max := -10.
	s2min := 10.
	for i := 0; i < len(states); i++ {
		if states[i][0] > s1max {
			s1max = states[i][0]
		}
		if states[i][0] < s1min {
			s1min = states[i][0]
		}
		if states[i][1] > s2max {
			s2max = states[i][1]
		}
		if states[i][1] < s2min {
			s2min = states[i][1]
		}
	}
	gPars.s1max = s1max
	gPars.s1min = s1min
	gPars.s2max = s2max
	gPars.s2min = s2min
	return
}

func gaussianFeatureSpace(state []float64) []float64 {
	newState := make([]float64, len(state))
	newState[0] = (state[0] - gPars.s1min) / (gPars.s1max - gPars.s1min)
	newState[1] = (state[1] - gPars.s2min) / (gPars.s2max - gPars.s2min)
	feature := make([]float64, 11)
	feature[0] = 1.
	feature[1] = math.Exp(-math.Pow((newState[0]-0.0), 2.) / (2 * math.Pow(0.25, 2)))
	feature[2] = math.Exp(-math.Pow((newState[0]-0.25), 2.) / (2 * math.Pow(0.25, 2)))
	feature[3] = math.Exp(-math.Pow((newState[0]-0.5), 2.) / (2 * math.Pow(0.25, 2)))
	feature[4] = math.Exp(-math.Pow((newState[0]-0.75), 2.) / (2 * math.Pow(0.25, 2)))
	feature[5] = math.Exp(-math.Pow((newState[0]-1.0), 2.) / (2 * math.Pow(0.25, 2)))
	feature[6] = math.Exp(-math.Pow((newState[1]-0.0), 2.) / (2 * math.Pow(0.25, 2)))
	feature[7] = math.Exp(-math.Pow((newState[1]-0.25), 2.) / (2 * math.Pow(0.25, 2)))
	feature[8] = math.Exp(-math.Pow((newState[1]-0.5), 2.) / (2 * math.Pow(0.25, 2)))
	feature[9] = math.Exp(-math.Pow((newState[1]-0.75), 2.) / (2 * math.Pow(0.25, 2)))
	feature[10] = math.Exp(-math.Pow((newState[1]-1.0), 2.) / (2 * math.Pow(0.25, 2)))
	return feature
}

// Feature spaces are determined by featSpace
// 1 = Second order polynomial
// 2 = Gaussian RBF
// 3 = Linear
func getFeatureSpace(state []float64, featSpace int) []float64 {
	feat := make([]float64, 0)
	if featSpace == 1 {
		feat = make([]float64, 6)
		feat[0] = 1.
		feat[1] = state[0]
		feat[2] = state[1]
		feat[3] = state[0] * state[1]
		feat[4] = state[0] * state[0]
		feat[5] = state[1] * state[1]
	} else if featSpace == 2 {
		feat = gaussianFeatureSpace(state)
	} else if featSpace == 3 {
		feat = make([]float64, 3)
		feat[0] = 1.
		feat[1] = state[0]
		feat[2] = state[1]
	}
	return feat
}

func multVec(vec []float64, factor float64) []float64 {
	n := len(vec)
	vec2 := make([]float64, n)
	for i := 0; i < n; i++ {
		vec2[i] = vec[i] * factor
	}
	return vec2
}

func divVec(vec []float64, factor float64) []float64 {
	n := len(vec)
	vec2 := make([]float64, n)
	for i := 0; i < n; i++ {
		vec2[i] = vec[i] / factor
	}
	return vec2
}

func addVec(vec1, vec2 []float64) []float64 {
	n := len(vec1)
	m := len(vec2)
	if n != m {
		panic("Error in 'addVec'. Vectors must be of the same length")
	}
	vec3 := make([]float64, n)
	for i := 0; i < n; i++ {
		vec3[i] = vec1[i] + vec2[i]
	}
	return vec3
}

func subVec(vec1, vec2 []float64) []float64 {
	n := len(vec1)
	m := len(vec2)
	if n != m {
		panic("Error in 'subVec'. Vectors must be of the same length")
	}
	vec3 := make([]float64, n)
	for i := 0; i < n; i++ {
		vec3[i] = vec1[i] - vec2[i]
	}
	return vec3
}

func dotProd(vec1, vec2 []float64) float64 {
	n := len(vec1)
	m := len(vec2)
	if n != m {
		panic("Error in 'dotProd'. Vectors must be of the same length")
	}
	result := 0.0
	for i := 0; i < n; i++ {
		result += vec1[i] * vec2[i]
	}
	return result
}

func getPsiElement(state, beta []float64, discount float64, pParamsGen, pParams policyParams, tParams transitionParams, nStates int, featSpace int, psiLoc *[]float64) {
	var action int
	var prob float64
	var probGen float64
	featDim := len(beta)
	source := rand.NewSource(time.Now().UnixNano())
	generator := rand.New(source)
	phiState := getFeatureSpace(state, featSpace)
	phiNext := make([]float64, featDim)
	nextState := make([]float64, 2)
	reward := 0.0
	num := make([]float64, featDim)
	denom := 0.0
	for i := 0; i < nStates; i++ {
		action, prob = policy(state, pParams, generator)
		probGen = getProb(state, action, pParamsGen)
		nextState = transition(state, action, tParams, generator)
		phiNext = getFeatureSpace(nextState, featSpace)
		reward = getReward(nextState, action)
		num = addVec(num, subVec(multVec(phiNext, discount), phiState))
		denom = denom + (prob/probGen)*math.Pow((reward+discount*dotProd(phiNext, beta)-dotProd(phiState, beta)), 2)
	}
	*psiLoc = divVec(num, denom)
	return
}

/* func getPsi(states [][]float64, beta []float64, discount float64, pParamsGen, pParams policyParams, tParams transitionParams, nStates int, featSpace int) [][]float64 {
	n := len(states)
	var wg sync.WaitGroup
	featDim := len(beta)
	psi := make([][]float64, n, n)
	for i := 0; i < n; i++ {
		psi[i] = make([]float64, featDim, featDim)
	}
	for i := 0; i < n; i++ {
		wg.Add(1)
		go getPsiElement(states[i], beta, discount, pParamsGen, pParams, tParams, nStates, featSpace, &psi[i], &wg)
	}
	wg.Wait()
	return psi
} */

func getPsi(states [][]float64, beta []float64, discount float64, pParamsGen, pParams policyParams, tParams transitionParams, nStates int, featSpace int) [][]float64 {
	n := len(states)
	featDim := len(beta)
	psi := make([][]float64, n, n)
	for i := 0; i < n; i++ {
		psi[i] = make([]float64, featDim, featDim)
	}
	numRoutines := 100
	var ch = make(chan int, n) // This number 50 can be anything as long as it's larger than xthreads
	var wg sync.WaitGroup
	wg.Add(numRoutines)
	for i := 0; i < numRoutines; i++ {
		go func() {
			for {
				a, ok := <-ch
				if !ok { // if there is nothing to do and the channel has been closed then end the goroutine
					wg.Done()
					return
				}
				getPsiElement(states[a], beta, discount, pParamsGen, pParams, tParams, nStates, featSpace, &psi[a]) // do the thing
			}
		}()
	}

	for i := 0; i < n; i++ {
		ch <- i // add i to the queue
	}

	close(ch) // This tells the goroutines there's nothing else to do
	wg.Wait() // Wait for the threads to finish

	return psi
}

func test(vec *[]float64) {
	*vec = []float64{0.1, 0.2}
}

// PyGetPsi calculates the Godambe Weights
//export PyGetPsi
func PyGetPsi(statesVec, beta, polParsGen, polPars, theta1, theta2 []float64, cSd1, cSd2 C.double, cNStates, cFeatSpace C.int, cDiscount C.double, cMaxProcs C.int) uintptr {

	runtime.GOMAXPROCS(int(cMaxProcs))

	numRows := len(statesVec) / 2
	stateMat := make([][]float64, numRows)

	for i := 0; i < numRows; i++ {
		stateMat[i] = []float64{statesVec[2*i], statesVec[2*i+1]}
	}

	nStates := int(cNStates)
	featSpace := int(cFeatSpace)
	discount := float64(cDiscount)
	sd1 := float64(cSd1)
	sd2 := float64(cSd2)

	if featSpace == 2 {
		setGaussianParams(stateMat, &gPars)
	}

	tParams := createTransPars(theta1, theta2, sd1, sd2)
	pParamsGen := createPolicyPars(polParsGen)
	pParams := createPolicyPars(polPars)

	psi := getPsi(stateMat, beta, discount, pParamsGen, pParams, tParams, nStates, featSpace)

	results := make([]float64, 0, numRows*len(psi[0]))
	for i := 0; i < numRows; i++ {
		results = append(results, psi[i]...)
	}

	return uintptr(unsafe.Pointer(&results[0]))

}

func getNewWeightElement(state, beta []float64, discount float64, pParams policyParams, tParams transitionParams, nStates int, featSpace int, weightLoc *float64) {
	var action int
	featDim := len(beta)
	source := rand.NewSource(time.Now().UnixNano())
	generator := rand.New(source)
	phiState := getFeatureSpace(state, featSpace)
	phiNext := make([]float64, featDim)
	nextState := make([]float64, 2)
	reward := 0.0
	denom := 0.0
	for i := 0; i < nStates; i++ {
		action, _ = policy(state, pParams, generator)
		nextState = transition(state, action, tParams, generator)
		phiNext = getFeatureSpace(nextState, featSpace)
		reward = getReward(nextState, action)
		denom = denom + (reward + discount*dotProd(phiNext, beta) - dotProd(phiState, beta))
	}
	*weightLoc = denom / float64(nStates)
	return
}

func getNewWeights(states [][]float64, beta []float64, discount float64, pParams policyParams, tParams transitionParams, nStates int, featSpace int) []float64 {
	n := len(states)
	weights := make([]float64, n, n)
	numRoutines := 100
	var ch = make(chan int, n) // This number 50 can be anything as long as it's larger than xthreads
	var wg sync.WaitGroup
	wg.Add(numRoutines)
	for i := 0; i < numRoutines; i++ {
		go func() {
			for {
				a, ok := <-ch
				if !ok { // if there is nothing to do and the channel has been closed then end the goroutine
					wg.Done()
					return
				}
				getNewWeightElement(states[a], beta, discount, pParams, tParams, nStates, featSpace, &weights[a]) // do the thing
			}
		}()
	}

	for i := 0; i < n; i++ {
		ch <- i // add i to the queue
	}

	close(ch) // This tells the goroutines there's nothing else to do
	wg.Wait() // Wait for the threads to finish

	return weights
}

// PyGetWeights calculates the Godambe Weights
//export PyGetWeights
func PyGetWeights(statesVec, beta, polPars, theta1, theta2 []float64, cSd1, cSd2 C.double, cNStates, cFeatSpace C.int, cDiscount C.double, cMaxProcs C.int) uintptr {

	runtime.GOMAXPROCS(int(cMaxProcs))

	numRows := len(statesVec) / 2
	stateMat := make([][]float64, numRows)

	for i := 0; i < numRows; i++ {
		stateMat[i] = []float64{statesVec[2*i], statesVec[2*i+1]}
	}

	nStates := int(cNStates)
	featSpace := int(cFeatSpace)
	discount := float64(cDiscount)
	sd1 := float64(cSd1)
	sd2 := float64(cSd2)

	if featSpace == 2 {
		setGaussianParams(stateMat, &gPars)
	}

	tParams := createTransPars(theta1, theta2, sd1, sd2)
	pParams := createPolicyPars(polPars)

	weights := getNewWeights(stateMat, beta, discount, pParams, tParams, nStates, featSpace)

	return uintptr(unsafe.Pointer(&weights[0]))

}

// New weight function with pi/mu
//
//
//

func getPMWeightElement(state, beta []float64, discount float64, pParams, pParamsGen policyParams, tParams transitionParams, nStates int, featSpace int, weightLoc *float64) {
	var action int
	var probGen float64
	var prob float64
	featDim := len(beta)
	source := rand.NewSource(time.Now().UnixNano())
	generator := rand.New(source)
	phiState := getFeatureSpace(state, featSpace)
	phiNext := make([]float64, featDim)
	nextState := make([]float64, 2)
	reward := 0.0
	denom := 0.0
	weight := 0.0
	for i := 0; i < nStates; i++ {
		action, probGen = policy(state, pParamsGen, generator)
		prob = getProb(state, action, pParams)
		nextState = transition(state, action, tParams, generator)
		phiNext = getFeatureSpace(nextState, featSpace)
		reward = getReward(nextState, action)
		weight = weight + (prob/probGen)*(reward+discount*dotProd(phiNext, beta)-dotProd(phiState, beta))
		denom = denom + (prob / probGen)
	}
	*weightLoc = weight / float64(nStates)
	return
}

func getPMWeights(states [][]float64, beta []float64, discount float64, pParams, pParamsGen policyParams, tParams transitionParams, nStates int, featSpace int) []float64 {
	n := len(states)
	weights := make([]float64, n, n)
	numRoutines := 100
	var ch = make(chan int, n) // This number 50 can be anything as long as it's larger than xthreads
	var wg sync.WaitGroup
	wg.Add(numRoutines)
	for i := 0; i < numRoutines; i++ {
		go func() {
			for {
				a, ok := <-ch
				if !ok { // if there is nothing to do and the channel has been closed then end the goroutine
					wg.Done()
					return
				}
				getPMWeightElement(states[a], beta, discount, pParams, pParamsGen, tParams, nStates, featSpace, &weights[a]) // do the thing
			}
		}()
	}

	for i := 0; i < n; i++ {
		ch <- i // add i to the queue
	}

	close(ch) // This tells the goroutines there's nothing else to do
	wg.Wait() // Wait for the threads to finish

	return weights
}

// PyGetPMWeights calculates the Godambe Weights
//export PyGetPMWeights
func PyGetPMWeights(statesVec, beta, polPars, polParsGen, theta1, theta2 []float64, cSd1, cSd2 C.double, cNStates, cFeatSpace C.int, cDiscount C.double, cMaxProcs C.int) uintptr {

	runtime.GOMAXPROCS(int(cMaxProcs))

	numRows := len(statesVec) / 2
	stateMat := make([][]float64, numRows)

	for i := 0; i < numRows; i++ {
		stateMat[i] = []float64{statesVec[2*i], statesVec[2*i+1]}
	}

	nStates := int(cNStates)
	featSpace := int(cFeatSpace)
	discount := float64(cDiscount)
	sd1 := float64(cSd1)
	sd2 := float64(cSd2)

	if featSpace == 2 {
		setGaussianParams(stateMat, &gPars)
	}

	tParams := createTransPars(theta1, theta2, sd1, sd2)
	pParams := createPolicyPars(polPars)
	pParamsGen := createPolicyPars(polParsGen)

	weights := getPMWeights(stateMat, beta, discount, pParams, pParamsGen, tParams, nStates, featSpace)

	return uintptr(unsafe.Pointer(&weights[0]))

}

var gPars gaussianParams

func main() {

	theta1 := []float64{0, -0.75, 0, 0.25, 0, 1.5, 0}
	theta2 := []float64{0, 0, 0.75, 0.25, 0, 0, -1.5}
	sd1 := 0.25
	sd2 := 0.25

	tParams := createTransPars(theta1, theta2, sd1, sd2)

	polPars := []float64{0., 0., 0., 0.}
	state := []float64{1., 1.}
	pParams := createPolicyPars(polPars)
	//source := rand.NewSource(0)
	source := rand.NewSource(time.Now().UnixNano())
	generator := rand.New(source)

	var action int
	//var prob float64

	//action, prob = policy(state, pParams, generator)

	stateAccu := []float64{0., 0.}
	for i := 0; i < 100000; i++ {
		nextState := transition(state, 1, tParams, generator)
		stateAccu = addVec(stateAccu, nextState)
	}
	stateAccu = divVec(stateAccu, 100000.)

	if stateAccu[0] > 1.01 || stateAccu[0] < .99 {
		fmt.Println("Transition function not working")
	}
	if stateAccu[1] > -0.49 || stateAccu[1] < -0.51 {
		fmt.Println("Transition function not working")
	}

	stateAccu = []float64{0., 0.}
	for i := 0; i < 100000; i++ {
		nextState := transition(state, 0, tParams, generator)
		stateAccu = addVec(stateAccu, nextState)
	}
	stateAccu = divVec(stateAccu, 100000.)

	if stateAccu[0] > -0.49 || stateAccu[0] < -0.51 {
		fmt.Println("Transition function not working")
	}
	if stateAccu[1] > 1.01 || stateAccu[1] < .99 {
		fmt.Println("Transition function not working")
	}

	state = []float64{-2., 3.}
	stateAccu = []float64{0., 0.}
	for i := 0; i < 100000; i++ {
		nextState := transition(state, 1, tParams, generator)
		stateAccu = addVec(stateAccu, nextState)
	}
	stateAccu = divVec(stateAccu, 100000.)

	if stateAccu[0] > -2.99 || stateAccu[0] < -3.01 {
		fmt.Println("Transition function not working")
	}
	if stateAccu[1] > -3.74 || stateAccu[1] < -3.76 {
		fmt.Println("Transition function not working")
	}
	stateAccu = []float64{0., 0.}
	for i := 0; i < 100000; i++ {
		nextState := transition(state, 0, tParams, generator)
		stateAccu = addVec(stateAccu, nextState)
	}
	stateAccu = divVec(stateAccu, 100000.)

	if stateAccu[0] > 0.01 || stateAccu[0] < -0.01 {
		fmt.Println("Transition function not working")
	}
	if stateAccu[1] > 0.76 || stateAccu[1] < 0.74 {
		fmt.Println("Transition function not working")
	}

	polPars = []float64{1., -1.5, .5, 1.0}
	pParams = createPolicyPars(polPars)
	state = []float64{1., 1.}

	actAccu := 0
	actProb := 0.
	prob1 := 0.
	prob1 = getProb(state, 1, pParams)
	for i := 0; i < 100000; i++ {
		action, _ = policy(state, pParams, generator)
		actAccu += action
	}
	actProb = float64(actAccu) / 100000
	if actProb > prob1+0.01 || actProb < prob1-0.01 {
		fmt.Println("Transition function not working")
	}

	fmt.Println(actProb)
	fmt.Println(prob1)
	//fmt.Println(stateAccu)

	state = []float64{-12., 13.}
	stateAccu = []float64{0., 0.}
	for i := 0; i < 100000; i++ {
		nextState := transition(state, 1, tParams, generator)
		stateAccu = addVec(stateAccu, nextState)
	}
	stateAccu = divVec(stateAccu, 100000.)

	if stateAccu[0] < -48.01 || stateAccu[0] > -47.99 {
		fmt.Println("Transition function not working")
	}
	if stateAccu[1] < -48.76 || stateAccu[1] > -48.74 {
		fmt.Println("Transition function not working")
	}
	fmt.Println(stateAccu)

	fmt.Println(dotProd([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}))

	beta := []float64{4.75316192, -0.50924432, 0.4022078, -0.23419914, 1.05442959,
		0.72264927}
	state = []float64{1., 1.}
	discount := 0.9
	polParsGen := []float64{0., 0., 0., 0.}
	pParamsGen := createPolicyPars(polParsGen)
	polPars = []float64{-10., 50., -30., 0.}
	pParams = createPolicyPars(polPars)
	psi := []float64{0., 0., 0., 0., 0., 0.}

	getPsiElement(state, beta, discount, pParamsGen, pParams, tParams, 100000, 1, &psi)

	fmt.Println(psi)

	weight := 0.0

	getNewWeightElement(state, beta, discount, pParams, tParams, 100000, 1, &weight)

	fmt.Println(weight)

}

// go build -o getValues.so -buildmode=c-shared getValues.go
