# valueIterationAgents.py
# -----------------------
##
import mdp, util
import sys

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp = None, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbabilities(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        if (self.mdp != None):
            self.doValueIteration()

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        #"*** YOUR CODE STARTS HERE ***"

        #util.raiseNotDefined()

        """
          This function is later used in doValueIteration and computeActionFromValues
          So we declare it beforehand
        """
	qvalue1 = 0

	for nextState, prob in self.mdp.getTransitionStatesAndProbabilities(state, action):
		qvalue1 += prob * (self.mdp.getReward(state,action,nextState) + (self.discount * self.values[nextState]))
        
        #"*** YOUR CODE FINISHES HERE ***"
        
	return qvalue1
    

    def doValueIteration (self):
        # Write value iteration code here

        print "Iterations: ", self.iterations
        print "Discount: ", self.discount
        states = self.mdp.getStates()
        maxDelta1 = float("-inf")


        #"*** YOUR CODE STARTS HERE ***"

        # Your code should include the implementation of value iteration
        # At the end it should show in the terminal the number of states considered in self.values and
        # the Delta between the last two iterations

	qvalue2 = 0

	for i in range (0, self.iterations):
		N = util.Counter()
		for state in states:
			if self.mdp.isTerminal(state):
				N[state] = 0
			else:
				Q = util.Counter()
				for action in self.mdp.getPossibleActions(state):
					Q[action] = self.computeQValueFromValues(state, action)
				N[state] = Q[Q.argMax()]
		self.values = N

	print "MaxDelta: ", maxDelta1
        
        #util.raiseNotDefined()
        #"*** YOUR CODE FINISHES HERE ***"
        
    def setMdp( self, mdp):
        """
          Set an mdp.
        """
        self.mdp = mdp
        self.doValueIteration()

    def setDiscount( self, discount):
        """
          Set a discount
        """
        self.discount = discount

    def setIterations( self, iterations):
        """
          Set a number of iterations
        """
        self.iterations = iterations
       
       
    def getValue(self, state):
        """
          Return the value of the state
        """
        return self.values[state]
        

    def showPolicy( self ):

        """
          Print the policy
        """
        
        states = self.mdp.getStates()
        for state in states:
            print "Policy\n", state, self.getPolicy(state)


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        #"*** YOUR CODE STARTS HERE ***"

	if self.mdp.isTerminal(state):
		return None

	maxDelta2, policy = float("-inf"), None
	qvalue3 = 0

	for action in self.mdp.getPossibleActions(state):
		qvalue3 = self.computeQValueFromValues(state, action)
		if qvalue3 > maxDelta2:
			maxDelta2 = qvalue3
			policy = action

	return policy

        #util.raiseNotDefined()

        #"*** YOUR CODE FINISHES HERE ***"

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getPolicy(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getAction(state)

    
    def getQValue(self, state, action):
        "Returns the Q value."        
        return self.computeQValueFromValues(state, action)

    def getPartialPolicy(self, stateL):
        "Returns the partial policy at the state. Random for unkown states"        
        state,state_names = self.mdp.stateToHigh(stateL)
        if self.mdp.isKnownState(state):
            return self.computeActionFromValues(state)
        else:
            # random action
            return util.random.choice(stateL.getLegalActions()) 
