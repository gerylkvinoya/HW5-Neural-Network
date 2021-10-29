##
# NeuralNets Agent
# CS 421
#
# Authors: Geryl Vinoya and Morgan Connor
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
        super(AIPlayer,self).__init__(inputPlayerId, "NeuralNets")
    
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
        stateList = []
        nodeList = []

        #for each move, get the resulting gamestate if we make that move and add it to the list
        for move in allMoves:

            if move.moveType == "END_TURN":
                continue

            newState = getNextState(currentState, move)
            stateList.append(newState)

            node = {
                'move' : move,
                'state' : newState,
                'depth' : 1,
                'eval' : self.utility(newState),
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
        #method templaste, not implemented
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
            return -1

        for i in range(len(weights)):
            output += weights[i] * inputs[i]
        
        return output

    #getOutput
    #
    #Description: get the sum of all the neurons in the hidden layer to apply to the output layer
    #
    #Parameters:
    #   inp - list of initial inputs
    #   hiddenWeights - list of initial weights (should be 40)
    #   outputWeights - list of output weights (should be 9)
    #   
    #
    #return: the output of the network
    def getOutput(self, inp, hiddenWeights, outputWeights):
        inputs = []
        inputs.append(1) #append 1 as the bias in the list in the first element

        #add all other inputs in the list 
        #inputs list looks like this: [bias, input1, input2, input3, input4]
        for input in inp:
            inputs.append(input)

        #list of numbers after using sig function (should be of size 8)
        activationList = []

        #get every 5 elements of the weights and put that in list to use self.activateNeuron on
        #len(hiddenWeights) should at least be divisible by 5; 5 weights represent a neuron
        for i in range(0, len(hiddenWeights), 5):
            activation = self.activateNeuron([hiddenWeights[i], hiddenWeights[i+1], 
                hiddenWeights[i+2], hiddenWeights[i+3], hiddenWeights[i+4]], inputs)

            activationList.append(self.sig(activation))

        #NEED TO TEST
        #NEED TO TEST
        #NEED TO TEST
        return self.activateNeuron(activationList, outputWeights)
            



        


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



    def testNeuralNetwork(self):
        player = AIPlayer(0)
        hiddenLayer = player.initWeights(40)
        outputLayer = player.initWeights(9)
        num = player.sig(1)



