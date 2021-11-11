##
# NeuralNets Agent
# CS 421
#
# Authors: Geryl Vinoya and Morgan Connor
#
# Sources Used:
#   https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
#   Dr. Nuxoll's slides
##
import random
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *
from typing import Dict, List
import unittest
import numpy as np


##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "NeuralNets_vinoya21_morganco23")
        self.gameStateList = []
        self.trainingAI = False
        self.hiddenLayer = [0.8219066024454105, -0.9835603916085868, -0.2550000074221794, 0.3467017949416797, -0.02979686303799146, -1.2951386956179647, -0.60743875417383, -0.5285435630126529, 0.224663263433739, -0.7166697958721399, -0.5347138283025262, 0.6657639408600032, -0.5003754218219909, 0.7614368749632836, 0.5181324637140722, 0.36093235343347496, 0.8078031140164182, 0.6563292688861689, 0.45174890922102673, 0.6222864061210843, 0.3166886920709504, -0.8966685193796649, -0.12469299684251174, 0.5971298481617191, 0.053414100705435306, -0.0642779684399614, -0.569434575675846, 0.13336878933017782, -0.7176503730989361, -0.9206912584312159, -1.1488020720771956, 
                            0.07692751116150676, 0.1929331075653529, 0.6413610608306726, 0.055086218325451235, 0.5370691053442583, 0.24376132365065858, 0.2916910784257264, -0.5403330269533996, -0.9376172778499439]
        self.outputLayer = [-0.03930064843838012, -0.5254829477702588, 1.464679348174604, 0.46984391088405447, 0.7685979336638357, 0.17438706411939478, 0.5126261205000846, 0.7991301699285988, -1.0728201864278921]
        #backups
        #hiddenLayer = [0.0846823655533066, 0.777402240228925, -0.4952711178922091, 0.7955426985421822, -0.34047766832833126, -0.6341186845957503, -0.48334718300436874, 0.10105274208395049, -1.5051809801497023, 0.35095739282477867, 0.8238040151874945, 0.7155152562465714, -1.0780880866926819, -0.7280217514583914, 0.27416790915727185, 0.7521548860007751, 0.33071524481196446, 0.9220924381251523, -0.8608065118060985, 0.6534344669672828, 0.6453022586477051, -0.27462492395406896, -0.10778443651707982, 1.1535460021655561, 0.5604186039295797, -0.6982549905496046, 1.9129618424120673, -0.7351593220516571, -0.5187265939383353, -0.08990677298086003, -0.2652465608771749, 0.5711741776232789, 0.5844182374202264, 0.5511194795047651, -0.5366756171486609, 0.0707750954526135, -0.33615638312374274, -2.0838075623321055, -0.9745750738278243, 0.2817166832292269, -0.28221130399789196, 1.092123124450522, -1.419826191082132, -0.3811764596668977, -3.2896959458827664, -1.448114723199332, -0.8477925354755759, -0.3544621772659615, 0.407958499364358, -0.3683205168012522, 2.505467035404279, 0.04043130101561472, -0.962495686580644, -2.2243079409110877]
        #outputLayer = 
    
    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        self.gameStateList = []
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):

        #self.move = Move
        #self.nextState = Gamestate
        #self.depth = 1
        #self.eval = Utility + self.depth
        #self.parent = None

        #create lists of all the moves and gameStates
        allMoves = listAllLegalMoves(currentState)
        self.gameStateList.append(currentState)
        stateList = []
        nodeList = []

        #for each move, get the resulting gamestate if we make that move and add it to the list
        for move in allMoves:

            if move.moveType == "END_TURN":
                continue

            newState = getNextState(currentState, move)

            node = None

            if self.trainingAI:
                stateList.append(newState)

                node = {
                    'move' : move,
                    'state' : newState,
                    'depth' : 1,
                    'eval' : self.utility(newState),
                    'parent': currentState
                }
                
            else:
                node = {
                    'move' : move,
                    'state' : newState,
                    'depth' : 1,
                    'eval' : self.neuralUtility(newState),
                    'parent': currentState
                }
            
            nodeList.append(node)
        
        #get the move with the best eval through the nodeList
        highestUtil = self.bestMove(nodeList)


        #return the move with the highest evaluation
        return highestUtil['move']

    
    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):

        if self.trainingAI:
            print("Game over! Training AI.")
            
            player = AIPlayer(0)
            #will use 8 inputs
            #6 hidden nodes (9 weights per node = 54)
            #1 output node (7 weights)

            testGameStates = self.gameStateList

            random.shuffle(testGameStates)

            #hiddenLayer = player.initWeights(54)
            #outputLayer = player.initWeights(7)
        

            keepGoing = True

            while keepGoing:
                #this block of code represents one epoch
                errorSum = 0

                for state in testGameStates:
                    #get the inputs for the current gamestate
                    inputs = player.calculateInputs(state)

                    #use the actual utility function for expected
                    expected = player.utility(state)

                    newWeights = player.backPropagate(inputs, expected, self.hiddenLayer, self.outputLayer)

                    #add the absolute value of the error to the average error of the epoch
                    absError = abs(newWeights[2])
                    errorSum += absError

                    #update the new layers
                    self.hiddenLayer = newWeights[0]
                    self.outputLayer = newWeights[1]

                avgError = errorSum/len(testGameStates)

                print("Average Error: " + str(avgError))
                if avgError < 0.05:
                    keepGoing = False
                    print(self.hiddenLayer)
                    print(self.outputLayer)

        pass

    ##
    #utility
    #Description: examines GameState object and returns a heuristic guess of how
    #               "good" that game state is on a scale of 0 to 1
    #
    #               a player will win if his opponent’s queen is killed, his opponent's
    #               anthill is captured, or if the player collects 11 units of food
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: the "guess" of how good the game state is
    ##
    def utility(self, currentState):
        WEIGHT = 10 #weight value for moves

        #will modify this toRet value based off of gamestate
        toRet = 0

        #get my id and enemy id
        me = currentState.whoseTurn
        enemy = 1 - me

        #get the values of the anthill, tunnel, and foodcount
        myTunnel = getConstrList(currentState, me, (TUNNEL,))[0]
        myAnthill = getConstrList(currentState, me, (ANTHILL,))[0]
        myFoodList = getConstrList(currentState, 2, (FOOD,))
        enemyTunnel = getConstrList(currentState, enemy, (TUNNEL,))[0]

        #get my soldiers and workers
        mySoldiers = getAntList(currentState, me, (SOLDIER,))
        myWorkerList = getAntList(currentState, me, (WORKER,))

        #get enemy worker and queen
        enemyWorkerList = getAntList(currentState, enemy, (WORKER,))
        enemyQueenList = getAntList(currentState, enemy, (QUEEN,))

        for worker in myWorkerList:

            #if a worker is carrying food, go to tunnel
            if worker.carrying:
                tunnelDist = stepsToReach(currentState, worker.coords, myTunnel.coords)
                #anthillDist = stepsToReach(currentState, worker.coords, myAnthill.coords)

                #if tunnelDist <= anthillDist:
                toRet = toRet + (1 / (tunnelDist + (4 * WEIGHT)))
                #else:
                    #toRet = toRet + (1 / (anthillDist + (4 * WEIGHT)))

                #add to the eval if a worker is carrying food
                toRet = toRet + (1 / WEIGHT)

            #if a worker isn't carrying food, get to the food
            else:
                foodDist = 1000
                for food in myFoodList:
                    # Updates the distance if its less than the current distance
                    dist = stepsToReach(currentState, worker.coords, food.coords)
                    if (dist < foodDist):
                        foodDist = dist
                toRet = toRet + (1 / (foodDist + (4 * WEIGHT)))
        
        #try to get only 1 worker
        if len(myWorkerList) == 1:
            toRet = toRet + (2 / WEIGHT)
        

        #try to get only one soldier
        if len(mySoldiers) == 1:
            toRet = toRet + (WEIGHT * 0.2)
            enemyWorkerLength = len(enemyWorkerList)
            enemyQueenLength = len(enemyQueenList)
            
            #we want the soldier to go twoards the enemy tunnel/workers
            if enemyWorkerList:
                distToEnemyWorker = stepsToReach(currentState, mySoldiers[0].coords, enemyWorkerList[0].coords)
                distToEnemyTunnel = stepsToReach(currentState, mySoldiers[0].coords, enemyTunnel.coords)
                toRet = toRet + (1 / (distToEnemyWorker + (WEIGHT * 0.2))) + (1 / (distToEnemyTunnel + (WEIGHT * 0.5)))
            
            #reward the agent for killing enemy workers
            #try to kill the queen if enemy workers dead
            else:
                toRet = toRet + (2 * WEIGHT)
                if enemyQueenLength > 0:
                    enemyQueenDist = stepsToReach(currentState, mySoldiers[0].coords, enemyQueenList[0].coords)
                    toRet = toRet + (1 / (1 + enemyQueenDist))
            

            toRet = toRet + (1 / (enemyWorkerLength + 1)) + (1 / (enemyQueenLength + 1))

        #try to get higher food score
        foodCount = currentState.inventories[me].foodCount
        toRet = toRet + foodCount

        #set the correct bounds for the toRet
        toRet = 1 - (1 / (toRet + 1))
        if toRet <= 0:
            toRet = 0.01
        if toRet >= 1:
            toRet = 0.99

        return toRet

    #bestMove
    #
    #Description: goes through each node in a list and finds the one with the 
    #highest evaluation
    #
    #Parameters: nodeList - the list of nodes you want to find the best eval for
    #
    #return: the node with the best eval
    def bestMove(self, nodeList):
        bestNode = nodeList[0]
        for node in nodeList:
            if (node['eval'] > bestNode['eval']):
                bestNode = node

        return bestNode

    #initWeights
    #
    #Description: Initiate weights for the hidden and output layers
    #             40 in the hidden layer, 9 in the output layer
    #
    #Parameters: numWeights - how many weights to create
    #
    #return: list of weights created
    def initWeights(self, numWeights):
        weightList = []
        for i in range(numWeights):
            weightList.append(random.uniform(-1.0, 1.0))
        return weightList

    #sig
    #
    #Description: perform the sigmoidal activation function
    #             1/(1+ e^{-x})
    #
    #Parameters: num - number for x
    #
    #return: the output of the sigmoidal activation function
    def sig(self, num):
        return 1/(1 + np.exp(-num))

    #activateNeuron
    #
    #Description: activation function on a single neuron given 4 inputs
    #             should have the same amount of weights as inputs
    #
    #Parameters:
    #   inp - list of inputs
    #   weights - list of weights
    #   
    #return: the output of the neuron activation
    def activateNeuron(self, inp, weights):
        output = 0
        inputs = []
        inputs.append(1) #append 1 as the bias in the list in the first element

        #add all other inputs in the list 
        #inputs list looks like this: [bias, input1, input2, input3, input4]
        for input in inp:
            inputs.append(input)

        #check the length of the lists
        if(len(inputs) != len(weights)):
            print("length of inputs not equal to length of weights")
            print(inputs)
            print(weights)
            return -1

        for i in range(len(weights)):
            output += weights[i] * inputs[i]
        
        return output

    #getHiddenOutputList
    #
    #Description: get a list of all the outputs of the nodes in the hidden layer
    #
    #Parameters:
    #   inp - list of initial inputs
    #   hiddenWeights - list of initial weights (should be 40)
    #   
    #return: a list of the hidden outputs
    def getHiddenOutputList(self, inp, hiddenWeights):
        inputs = []
        #inputs.append(1) #append 1 as the bias in the list in the first element

        #add all other inputs in the list 
        #inputs list looks like this: [bias, input1, input2, input3, input4]
        for input in inp:
            inputs.append(input)


        #list of numbers after using sig function (should be of size 8)
        hiddenOutputList = []

        #changed from 5 to 9
        if (len(hiddenWeights) % 5) != 0:
            print("len(hiddenWeights) not divisible by 5")
            print(len(hiddenWeights))

        #get every 9 elements of the weights and put that in list to use self.activateNeuron on
        #len(hiddenWeights) should at least be divisible by 9; 9 weights represent a neuron
        for i in range(0, len(hiddenWeights), 5):
            activation = self.activateNeuron(inputs, [hiddenWeights[i], hiddenWeights[i+1], 
                hiddenWeights[i+2], hiddenWeights[i+3], hiddenWeights[i+4]])
                #, hiddenWeights[i+5], hiddenWeights[i+6], hiddenWeights[i+7], hiddenWeights[i+8]]

            hiddenOutputList.append(self.sig(activation))

        return hiddenOutputList

    #getOutput
    #
    #Description: get the sum of all the neurons in the hidden layer to apply to the output layer
    #
    #Parameters:
    #   inp - list of initial inputs
    #   hiddenWeights - list of initial weights (should be 40)
    #   outputWeights - list of output weights (should be 9)
    #   
    #return: the output of the network
    def getOutput(self, inp, hiddenWeights, outputWeights):
        inputs = []
        #inputs.append(1) #append 1 as the bias in the list in the first element

        #add all other inputs in the list 
        #inputs list looks like this: [bias, input1, input2, input3, input4]
        for input in inp:
            inputs.append(input)

        #list of numbers after using sig function (should be of size 8)
        sigList = []

        #grab the list of the hidden layer outputs
        #perform sig fn on all of them into a list
        hiddenOutputList = self.getHiddenOutputList(inp, hiddenWeights)

        #"activate" the output node to get the overall output
        return self.sig(self.activateNeuron(hiddenOutputList, outputWeights))
            
    #getErrorTerm
    #
    #Description: calculate the error term
    #             Err Term = (Err)*(output)(1-output)
    #
    #Parameters:
    #   error - the error
    #   actual - the actual output
    #   
    #return: the error term
    def getErrorTerm(self, error, actual):
        return (error)*(actual)*(1 - actual)
    
    #getHiddenNodeError
    #
    #Description: use the error term to calculate the error for each hidden node
    #             weight*err term
    #
    #Parameters:
    #   errTerm - the error term
    #   weights - the weights of the output node
    #
    #return: list of nodes' error
    def getHiddenNodeError(self, errTerm, weights):

        hiddenErrorList = []
        #start at 1 because index 0 should be the bias
        for i in range(1, len(weights), 1):
            hiddenErrorList.append(errTerm*weights[i])

        return hiddenErrorList

    #getHiddenNodeErrorTerms
    #
    #Description: use the error term to calculate the error for each hidden node
    #             weight*err term
    #
    #Parameters:
    #   hiddenOutputList - list of all the hidden nodes' output
    #   hiddenErrorList - list of all the nodes' errors
    #
    #return: list of nodes' error term
    def getHiddenNodeErrorTerms(self, hiddenOutputList, hiddenErrorList):
        if len(hiddenOutputList) != len(hiddenErrorList):
            print("Length of hidden output list and hidden error list are not equal")
            return
        
        hiddenErrorTermList = []
        for i in range(len(hiddenOutputList)):

            hiddenErrorTermList.append(self.getErrorTerm(hiddenErrorList[i], hiddenOutputList[i]))
        
        return hiddenErrorTermList 

    #adjustWeight
    #
    #Description: adjust a node's weight
    #
    #Parameters:
    #   weight - initial weight W
    #   errorTerm - error term delta
    #   input - for x_j
    #
    #return: new list of adjusted weights
    def adjustWeight(self, weight, errorTerm, input):
        alpha = 0.5
        
        return weight + (alpha*errorTerm*input)

    #backPropagate
    #
    #Description: perform back propagation on an entire system
    #
    #Parameters:
    #   inp - input
    #   hiddenWeights - weights in the hidden layer (should be 40)
    #   outputWeights - weights in the output later (should be 9)
    #
    #return: list of list of two new hidden and output weight lists
    def backPropagate(self, inp, target, hiddenWeights, outputWeights):

        

        #calculate output
        output = self.getOutput(inp, hiddenWeights, outputWeights)
        #calculate error term
        error = target - output
        outputErrorTerm = self.getErrorTerm(error, output)

        #calculate the error term for all the output nodes
        hiddenNodeErrorList = self.getHiddenNodeError(outputErrorTerm, outputWeights)
        hiddenOutputList = self.getHiddenOutputList(inp, hiddenWeights)
        hiddenErrorTerms = self.getHiddenNodeErrorTerms(hiddenOutputList, hiddenNodeErrorList)

        newHiddenWeights = []
        newOutputWeights = []

        #adds the bias to the beginning of a new input list
        #[bias, input1, input2, input3, input4]
        inputs = []
        inputs.append(1)
        for input in inp:
            inputs.append(input)

        #start with the hidden weights
        for i in range(0, len(hiddenWeights), 1):
            node = self.getNodeIndex(i)
            inputIndex = i % 5
            newHiddenWeights.append(self.adjustWeight(hiddenWeights[i], hiddenErrorTerms[node], inputs[inputIndex]))
        
        #insert a 1 at the beginning for the bias
        hiddenOutputList.insert(0, 1)
        for i in range(0, len(outputWeights), 1):
            newOutputWeights.append(self.adjustWeight(outputWeights[i], outputErrorTerm, hiddenOutputList[i]))

        

        return [newHiddenWeights, newOutputWeights, error]
        
    #getNodeIndex
    #
    #Description: given a number between 0 and 39,
    #             find what node the weight belongs to
    #
    #Parameters:
    #   num - index of the weight
    #
    #return: index of node this weight belongs to
    def getNodeIndex(self, num):
        #indexes 0 - 4 are in node 1, 5 - 9 are in node 2, etc...
        return int(num/5)


    def calculateInputs(self, currentState):
        
        #create empty list of float inputs
        inputs = [-1.0] * 4

        WEIGHT = 10 #weight value for moves

        #get my id and enemy id
        me = currentState.whoseTurn
        enemy = 1 - me

        #get the values of the anthill, tunnel, and foodcount
        myTunnel = getConstrList(currentState, me, (TUNNEL,))[0]
        myAnthill = getConstrList(currentState, me, (ANTHILL,))[0]
        myFoodList = getConstrList(currentState, 2, (FOOD,))
        enemyTunnel = getConstrList(currentState, enemy, (TUNNEL,))[0]

        #get my soldiers and workers
        mySoldiers = getAntList(currentState, me, (SOLDIER,))
        myWorkerList = getAntList(currentState, me, (WORKER,))

        #get enemy worker and queen
        enemyWorkerList = getAntList(currentState, enemy, (WORKER,))
        enemyQueenList = getAntList(currentState, enemy, (QUEEN,))

        workerUtil = 0.0

        for worker in myWorkerList:

            #if a worker is carrying food, go to tunnel
            if worker.carrying:
                tunnelDist = stepsToReach(currentState, worker.coords, myTunnel.coords)
                #anthillDist = stepsToReach(currentState, worker.coords, myAnthill.coords)

                #if tunnelDist <= anthillDist:
                workerUtil = workerUtil + (1 / (tunnelDist + (4 * WEIGHT)))
                #else:
                    #toRet = toRet + (1 / (anthillDist + (4 * WEIGHT)))

                #add to the eval if a worker is carrying food
                workerUtil = workerUtil + (1 / WEIGHT)

            #if a worker isn't carrying food, get to the food
            else:
                foodDist = 1000
                for food in myFoodList:
                    # Updates the distance if its less than the current distance
                    dist = stepsToReach(currentState, worker.coords, food.coords)
                    if (dist < foodDist):
                        foodDist = dist
                workerUtil = workerUtil + (1 / (foodDist + (4 * WEIGHT)))
        
        inputs[0] = workerUtil

        #try to get only 1 worker
        if len(myWorkerList) == 1:
            inputs[1] = (2 / WEIGHT)
        
        else:
            inputs[1] = 0.0
        
        soldierUtil = 0.0

        #try to get only one soldier
        if len(mySoldiers) == 1:
            soldierUtil = soldierUtil +  (WEIGHT * 0.2)
            enemyWorkerLength = len(enemyWorkerList)
            enemyQueenLength = len(enemyQueenList)
            


            #we want the soldier to go twoards the enemy tunnel/workers
            if enemyWorkerList:
                distToEnemyWorker = stepsToReach(currentState, mySoldiers[0].coords, enemyWorkerList[0].coords)
                distToEnemyTunnel = stepsToReach(currentState, mySoldiers[0].coords, enemyTunnel.coords)
                soldierUtil = soldierUtil + (1 / (distToEnemyWorker + (WEIGHT * 0.2))) + (1 / (distToEnemyTunnel + (WEIGHT * 0.5)))
            
            #reward the agent for killing enemy workers
            #try to kill the queen if enemy workers dead
            else:
                soldierUtil = soldierUtil + (2 * WEIGHT)
                if enemyQueenLength > 0:
                    enemyQueenDist = stepsToReach(currentState, mySoldiers[0].coords, enemyQueenList[0].coords)
                    soldierUtil = soldierUtil + (1 / (1 + enemyQueenDist))
            

            soldierUtil = soldierUtil + (1 / (enemyWorkerLength + 1)) + (1 / (enemyQueenLength + 1))

        inputs[2] = soldierUtil

        #try to get higher food score
        foodCount = currentState.inventories[me].foodCount
        inputs[3] = foodCount/11

        return inputs


    ####
    # list of inputs
    #
    # 0-number of workers: 0 or 1
    # 1-more than one worker 0(no) or 1(yes)
    # 2-is worker carrying food: 0(no) or 1(yes)
    # 3-worker distance from tunnel: 0(on the tunnel) to 1(10 turns away from the tunnel)
    # 4-number of soldiers we have: 0 or 1
    # 5-distance between soldier and enemy tunnel 0(on the enemy tunnel) to 1(10 turns away from enemy tunnel)
    # 6-distance between soldier and enemy worker(s) 0(next to enemy worker) to 1(10 turns away from enemy worker)
    # 7-number of enemy workers: 0(0) to 1(3 or more)

    #neuralUtility
    #
    #Description: use the given weights to find a new utility
    #
    #Parameters:
    #   currentState - state of the game
    #
    #return: float from 0...1 depending on how "good" a gamestate is
    def neuralUtility(self, currentState):
        inputs = self.calculateInputs(currentState)

        return self.getOutput(inputs, self.hiddenLayer, self.outputLayer)


class TestCreateNode(unittest.TestCase):

    #queens, anthills, and tunnels only
    def testUtilityBasic(self):
        player = AIPlayer(0)
        gameState = GameState.getBasicState()

        self.assertEqual(player.utility(gameState), 0.01)

    def testBestMove(self):
        player = AIPlayer(0)

        nodes = []

        #making node objects (only eval is used)
        for i in range(10):
            node = {
                'eval' : i,
            }
            nodes.append(node) 

        best = player.bestMove(nodes)

        self.assertEqual(best['eval'], 9)

    def testInitWeights(self):
        player = AIPlayer(0)
        hiddenLayer = player.initWeights(40)
        outputLayer = player.initWeights(9)

        for num in hiddenLayer:
            self.assertAlmostEqual(num, 0, delta=1)
        
        for num in outputLayer:
            self.assertAlmostEqual(num, 0, delta=1)
        
        self.assertEqual(40, len(hiddenLayer))
        self.assertEqual(9, len(outputLayer))

    def testSigmoid(self):
        player = AIPlayer(0)
        num = player.sig(1)
        self.assertAlmostEqual(num, 0.7310585786300049)

    def testActivateNeuron(self):
        player = AIPlayer(0)
        weights = [-0.5061572036196194, 0.11710044128933261, -0.5640861215824573, -0.6753749016864017, 0.20443410702909404]
        inputs = [1, 0, 0, 1]
        activation = player.activateNeuron(inputs, weights)
        self.assertAlmostEqual(activation, (-0.184622655301))
    
    def testGetOutputOneNeuron(self):
        player = AIPlayer(0)

        hiddenWeights = [-0.5061572036196194, 0.11710044128933261, -0.5640861215824573, -0.6753749016864017, 0.20443410702909404]
        outputWeights = [0.2395010298047295, -0.79178479274999917]
        inputs = [1, 0, 0, 1]

        output = player.getOutput(inputs, hiddenWeights, outputWeights)
        #CHECK to make sure this is the expected number
        self.assertAlmostEqual(output, 0.4700, delta=0.0001)

    def testGetOutputEightNeurons(self):
        player = AIPlayer(0)

        #40 weights for 8 neurons
        hiddenWeights = [0.3415, -0.4910, 0.7999, 0.1322, -0.9931, 
                         0.5132, -0.1122, -0.8483, 0.6340, 0.8888,
                         0.1342, -0.9348, -0.1234, 0.4333, -0.1222,
                         0.3937, -0.3882, 0.5555, 0.9294, 0.8726,
                         0.3947, 0.9673, 0.4872, -0.8366, -0.2838,
                         0.6333, -0.4522, 0.9983, 0.8272, 0.2333,
                         0.3344, -0.5523, -0.9101, 0.3710, 0.3999,
                         -0.1233, -0.3456, -0.3291, -0.9967, -0.8437]

        #9 weights for one output
        outputWeights = [0.2334, -0.2985, 0.9090, 0.7329, 0.1121,
                         0.1022, -0.5234, -0.6444, -0.7291]

        inputs = [1, 0, 0, 1]

        #round to 4 places, it's close enough after testing with 3 different sets of numbers
        aiOutput = round(player.getOutput(inputs, hiddenWeights, outputWeights), 4)

        #CHECK to make sure this is the expected number
        self.assertAlmostEqual(aiOutput, 0.6027, delta=0.001)

    def testGetErrorTerm(self):
        player = AIPlayer(0)
        #self.assertAlmostEqual(player.getErrorTerm(0, 0.4101), -0.0992, delta=0.0001)
        #self.assertAlmostEqual(player.getErrorTerm(1, 0.3820), -0.1457, delta=0.0001)
        self.assertAlmostEqual(player.getErrorTerm(-0.2650, 0.2650), -0.0516, delta=0.0001)

    def testGetHiddenNodeError(self):
        player = AIPlayer(0)

        errTerm = player.getErrorTerm(-0.2650, 0.2650)

        outputWeights = [0.2334, -0.2985, 0.9090, 0.7329, 0.1121,
                         0.1022, -0.5234, -0.6444, -0.7291]

        hiddenErrorList = player.getHiddenNodeError(errTerm, outputWeights)

        expectedList = [0.0154026, -0.0469044, -0.03781764, -0.00578436,
            -0.00527352, 0.02700744, 0.03325104, 0.03762156]

        for i in range(len(hiddenErrorList)):
            self.assertAlmostEqual(hiddenErrorList[i], expectedList[i], delta=0.0001)
    
    def testGetHiddenOutputList(self):
        player = AIPlayer(0)
        inp = [1, 0, 0, 1]
        hiddenWeights = [0.5061, 0.1171, -0.5640, -0.6753, 0.2044, 
                        0.1342, -0.4829, 0.8382, -0.3222, 0.0421]
        
        activationList = [0.8276, -0.3066]
        sigList = []
        for num in activationList:
            sigList.append(player.sig(num))

        self.assertEqual(player.getHiddenOutputList(inp, hiddenWeights), sigList)


    def testGetHiddenNodeErrorTerms(self):
        player = AIPlayer(0)

        inp = [1, 0, 0, 1]
        hiddenWeights = [0.5061, 0.1171, -0.5640, -0.6753, 0.2044, 
                        0.1342, -0.4829, 0.8382, -0.3222, 0.0421]

        #[0.695847223430284, 0.42394485650174163]
        hiddenOutputList = player.getHiddenOutputList(inp, hiddenWeights)
        

        hiddenErrorList = [0.0154, -0.0469]

        hiddenNodeErrorTermsList = player.getHiddenNodeErrorTerms(hiddenOutputList, hiddenErrorList)
        expectedList  = [0.0032, -0.0114]
        for i in range(len(hiddenErrorList)):
            self.assertAlmostEqual(hiddenNodeErrorTermsList[i], expectedList[i], delta=0.0001)

    def testAdjustWeight(self):
        player = AIPlayer(0)
        self.assertAlmostEqual(player.adjustWeight(0.1, 0.0075, 1), 0.1038, delta=0.0001)

    def testGetNodeIndex(self):
        player = AIPlayer(0)
        #changing to account for 9 weights
        self.assertEqual(player.getNodeIndex(3), 0)
        self.assertEqual(player.getNodeIndex(5), 0)
        self.assertEqual(player.getNodeIndex(14), 1)
        self.assertEqual(player.getNodeIndex(17), 1)
        self.assertEqual(player.getNodeIndex(23), 2)

    def testBackPropagate(self):
        #40 weights for 8 neurons
        hiddenWeights = [0.3415, -0.4910, 0.7999, 0.1322, -0.9931, 
                         0.5132, -0.1122, -0.8483, 0.6340, 0.8888,
                         0.1342, -0.9348, -0.1234, 0.4333, -0.1222,
                         0.3937, -0.3882, 0.5555, 0.9294, 0.8726,
                         0.3947, 0.9673, 0.4872, -0.8366, -0.2838,
                         0.6333, -0.4522, 0.9983, 0.8272, 0.2333,
                         0.3344, -0.5523, -0.9101, 0.3710, 0.3999,
                         -0.1233, -0.3456, -0.3291, -0.9967, -0.8437]

        #9 weights for one output
        outputWeights = [0.2334, -0.2985, 0.9090, 0.7329, 0.1121,
                         0.1022, -0.5234, -0.6444, -0.7291]




player = AIPlayer(0)
gameState = GameState.getBasicState()

#will use 8 inputs
#6 hidden nodes (9 weights per node = 54)
#1 output node (7 weights)

#some random numbers for the inputs
inputs = player.calculateInputs(gameState)

#print(player.utility(gameState))

#print(inputs)

hiddenLayer = player.initWeights(40)
outputLayer = player.initWeights(9)

#print(hiddenLayer)
#print(outputLayer)

#setting keepGoing to false because we don't need to backpropagate at the moment
keepGoing = False

while keepGoing:
    #this block of code represents one epoch
    errorSum = 0

    #use the actual utility function for expected
    expected = player.utility(gameState)
    newWeights = player.backPropagate(inputs, expected, hiddenLayer, outputLayer)

    #add the absolute value of the error to the average error of the epoch
    absError = abs(newWeights[2])
    errorSum += absError

    #update the new layers
    hiddenLayer = newWeights[0]
    outputLayer = newWeights[1]

    avgError = errorSum

    print("Average Error: " + str(avgError))
    if avgError < 0.0001:
        keepGoing = False
        print(hiddenLayer)
        print(outputLayer)


        

'''
#epoch series test
player = AIPlayer(0)
possibleInputs = [[0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1],
                    [0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 1],
                    [0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 1, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 1, 1],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 1, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1]]

#pick 10 random inputs
testInputs = random.sample(possibleInputs, 10)

#init random values
hiddenLayer = player.initWeights(40)
outputLayer = player.initWeights(9)

keepGoing = True
while keepGoing:

    #this block of code represents one epoch
    errorSum = 0
    for inp in testInputs:
        inputs = inp[0:4]
        expected = inp[4]
        newWeights = player.backPropagate(inputs, expected, hiddenLayer, outputLayer)

        #add the absolute value of the error to the average error of the epoch
        absError = abs(newWeights[2])
        errorSum += absError

        #update the new layers
        hiddenLayer = newWeights[0]
        outputLayer = newWeights[1]

    avgError = errorSum/len(testInputs)

    print("Average Error: " + str(avgError))
    if avgError < 0.05:
        keepGoing = False
'''
