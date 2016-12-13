# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        # V(s) = foreach Action get the max. Foreach Action do T(s,a,s')*[R(s,a,s')+&*V(s')].
        #                                                      T=possiblity         &=discount


        # Write value iteration code here
        for i in xrange(self.iterations):
            oldValues =self.values.copy()
            for s in self.mdp.getStates():
                qvalue = util.Counter()
                for a in self.mdp.getPossibleActions(s):
                    trans = util.Counter()
                    for s2,p in self.mdp.getTransitionStatesAndProbs(s,a):
                        trans[s2] = p * (self.mdp.getReward(s, a, s2) + self.discount * oldValues[s2])
                    qvalue[a] = trans.totalCount()
                self.values[s] = qvalue[qvalue.argMax()]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qvalue = util.Counter()
        for s2,p in self.mdp.getTransitionStatesAndProbs(state,action):
            qvalue[s2] = p * (self.mdp.getReward(state, action, s2) + self.discount * self.getValue(s2))

        return qvalue.totalCount()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None

        bestAction = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            bestAction[action] = self.computeQValueFromValues(state,action)
        return bestAction.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
