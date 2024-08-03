# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # ReflexAgent：根据对当前环境的感知，作出相应行动，其行动分为正反馈和负反馈两种
        # 我们需要对当前环境中的鬼怪和豆豆这两大因素进行量化

        # 正反馈： 优先考虑吃掉最近的豆豆
        if newFood.asList() != []:
            # 如果豆豆没有被吃完，考虑最近的豆豆
            nearest_dot = min(manhattanDistance(newPos, food) for food in newFood.asList())
            # 正反馈永远是正数，并且吃到隔壁的豆豆可以得10分，但是移动需要消耗1分，因此净赚9分
            # 此外，正反馈与距离成反比，因此我们考虑用一个反比例函数作为正反馈的分数
            postive_score = 9.0 / nearest_dot
        else:
            postive_score = 0
        # 负反馈： 我们需要考虑鬼怪对吃豆人的影响，思路仍然是找到最近的鬼怪
        # 将所有鬼怪的曼哈顿距离存入列表中
        ghost_dists = []
        for ghostState in newGhostStates:
            # 计算鬼怪与当前位置的曼哈顿距离
            ghost_dist = manhattanDistance(ghostState.configuration.pos, newPos)
            ghost_dists.append(ghost_dist)
        
        nearest_ghost = min(ghost_dists)
        negative_score = -10.0 / nearest_ghost if nearest_ghost != 0 else 0
        return successorGameState.getScore() + postive_score + negative_score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        maxVal, bestAction = -float('inf'), None
        for action in gameState.getLegalActions(0):
            # 0代表吃豆人, 求出所有可行的action中，哪一个是最优的
            value = self.getMin(gameState.generateSuccessor(0, action), 0, 1)
            if value is not None and value > maxVal:
                maxVal = value
                bestAction = action
        return bestAction

 
    # getMax主要是给吃豆人选择最佳的动作使用
    def getMax(self, gameState, depth=0, agentIndex=0):
        # 获得当前Agent的所有合法的下一步动作
        legalActions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(legalActions) == 0:
            # 到达搜索深度需要停止搜索，同样，如果接下来没有可行的行动也需要中止
            return self.evaluationFunction(gameState)
        # 初始化v为负无穷
        maxVal = -float('inf')
        # 通过对所用合法动作遍历，获得所有评价值和v的最大值
        for action in legalActions:
            # 接下来的动作是计算鬼怪的行动影响
            value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, 1)
            if value is not None and value > maxVal:
                maxVal = value
        return maxVal
        
    # getMin主要是计算鬼怪选择最坏影响的动作
    def getMin(self, gameState, depth=0, agentIndex=1):
        legalActions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(legalActions) == 0:
            return self.evaluationFunction(gameState)
        # 初始化v为正无穷
        minVal = float('inf')
        for action in legalActions:
            if agentIndex == gameState.getNumAgents() - 1:
                # 如果是最后一个鬼怪的agent。接下来需要计算吃豆人
                value = self.getMax(gameState.generateSuccessor(agentIndex, action), depth + 1, 0)
            else:
                value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            if value is not None and value < minVal:
                minVal = value
        return minVal



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxVal, bestAction = -float('inf'), None
        alpha, beta = -float('inf'), float('inf')
        for action in gameState.getLegalActions(0):
            # 0代表吃豆人, 求出所有可行的action中，哪一个是最优的
            value = self.getMin(gameState.generateSuccessor(0, action), 0, 1, alpha=alpha, beta=beta)
            if value is not None and value > maxVal:
                maxVal = value
                bestAction = action
            if value > beta:
                return value, action
            # 更新alpha值
            alpha = value if value > alpha else alpha
        return bestAction

 
    # getMax主要是给吃豆人选择最佳的动作使用
    def getMax(self, gameState, depth=0, agentIndex=0, alpha=-float('inf'), beta=float('inf')):
        # 获得当前Agent的所有合法的下一步动作
        legalActions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(legalActions) == 0:
            # 到达搜索深度需要停止搜索，同样，如果接下来没有可行的行动也需要中止
            return self.evaluationFunction(gameState)
        # 初始化v为负无穷
        maxVal = -float('inf')
        # 通过对所用合法动作遍历，获得所有评价值和v的最大值
        for action in legalActions:
            # 接下来的动作是计算鬼怪的行动影响
            value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, 1, alpha, beta)
            if value is not None and value > maxVal:
                maxVal = value
            # 剪枝部分
            if value > beta:
                return value
            # 更新alpha值
            alpha = value if value > alpha else alpha
        return maxVal
        
    # getMin主要是计算鬼怪选择最坏影响的动作
    def getMin(self, gameState, depth=0, agentIndex=1, alpha=-float('inf'), beta=float('inf')):
        legalActions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(legalActions) == 0:
            return self.evaluationFunction(gameState)
        # 初始化v为正无穷
        minVal = float('inf')
        for action in legalActions:
            if agentIndex == gameState.getNumAgents() - 1:
                # 如果是最后一个鬼怪的agent。接下来需要计算吃豆人
                value = self.getMax(gameState.generateSuccessor(agentIndex, action), depth + 1, 0, alpha, beta)
            else:
                value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta)
            if value is not None and value < minVal:
                minVal = value
            # 剪枝部分
            if value < alpha:
                return value
            # 更新beta值
            beta = value if value < beta else beta
        return minVal

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.getMax(gameState)[1]
    
    # 与Minimax一样，getMax主要是计算吃豆人的最佳行动
    def getMax(self, gameState, depth=0, agentIndex=0):
        legalActions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(legalActions) == 0:
            # 到达搜索深度需要停止搜索，同样，如果接下来没有可行的行动也需要中止
            return self.evaluationFunction(gameState), None
        max_val, bestAction = -float('inf'), None
        for action in legalActions:
            value = self.getExpect(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            if value > max_val:
                max_val = value
                bestAction = action
        return max_val, bestAction

    

    def getExpect(self, gameState, depth, agentIndex=1):
        # 默认ExpectiMax是计算鬼怪选择造成影响状态的效用值，即各种可能的效用值平均
        legalActions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(legalActions) == 0:
            # 到达搜索深度需要停止搜索，同样，如果接下来没有可行的行动也需要中止
            return self.evaluationFunction(gameState)
        # 获得当前鬼怪的所有可行操作，并进行遍历，求ExpectValue
        total = 0
        for action in legalActions:
            if agentIndex == gameState.getNumAgents() - 1:
                # 轮到吃豆人，只需要value即可
                value = self.getMax(gameState.generateSuccessor(agentIndex, action), depth + 1, 0)[0]
            else:
                value = self.getExpect(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            total += value
        # 计算平均期望值
        return total / len(legalActions)
                



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    # ReflexAgent：根据对当前环境的感知，作出相应行动，其行动分为正反馈和负反馈两种
    # 我们需要对当前环境中的鬼怪和豆豆这两大因素进行量化

    # 正反馈： 优先考虑吃掉最近的豆豆
    if newFood.asList() != []:
        # 如果豆豆没有被吃完，考虑最近的豆豆
        nearest_dot = min(manhattanDistance(newPos, food) for food in newFood.asList())
        # 正反馈永远是正数，并且吃到隔壁的豆豆可以得10分，但是移动需要消耗1分，因此净赚9分
        # 此外，正反馈与距离成反比，因此我们考虑用一个反比例函数作为正反馈的分数
        postive_score = 9.0 / nearest_dot
    else:
        postive_score = 0
    # 负反馈： 我们需要考虑鬼怪对吃豆人的影响，思路仍然是找到最近的鬼怪
    # 将所有鬼怪的曼哈顿距离存入列表中
    ghost_dists = []
    for ghostState in newGhostStates:
        # 计算鬼怪与当前位置的曼哈顿距离
        ghost_dist = manhattanDistance(ghostState.configuration.pos, newPos)
        ghost_dists.append(ghost_dist)
        nearest_ghost = min(ghost_dists)
        negative_score = -10.0 / nearest_ghost if nearest_ghost != 0 else 0
        return currentGameState.getScore() + postive_score + negative_score + sum(newScaredTimes)

# Abbreviation
better = betterEvaluationFunction
