import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""



    def __init__(self, env):
        super(LearningAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'blue'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.alpha = 0.2
        self.gamma = 0.9
        self.state = None
        self.action = None
        self.OLDstate = None
        self.NEWstate = None
        self.agent_reward = None
        self.storageRewards = 0  # just for testing
        self.countFail = 0
        self.QL = dict()  # storage the state and action here. (https://docs.python.org/2/library/stdtypes.html)
        self.exploration_rate = 0.1  # percentage of randomness
        self.Learning = True # choice QL ou random


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.action = None
        self.storageRewards = 0

    def qlvalue(self, state, action):  # return the Q value
        return self.QL.get((state, action), 20.0)

    def mapActions(self, state):  # return action from state
        return ['forward', 'left', 'right', None]

    def mapState(self, state):
        State = namedtuple("State", ["light", "next_waypoint"])
        return State(light=state['light'],
                     next_waypoint=self.planner.next_waypoint())

    def best_qvalue(self, state):
        Actions = self.mapActions(state)
        qv = - 999999999
        for action in Actions:
            if self.qlvalue(state, action) > qv:
                qv = self.qlvalue(state, action)
        return qv

    def QLupdate(self, reward, action, state, nextState):
        self.QL[(state, action)] = self.qlvalue(state, action) + self.alpha * (
            reward + (self.gamma * self.best_qvalue(nextState)) - self.qlvalue(state, action))

    def choiceRandomAction(self, state):
        Actions = self.mapActions(state)
        bestAction = random.choice(Actions)
        return bestAction


    def choiceAction(self, state):
        Actions = self.mapActions(state)
        bestAction = None
        QV = - 999999999

        rvalue = random.random()
        if rvalue < self.exploration_rate:  # do a random action
            bestAction= random.choice(Actions)
        else:
            for action in Actions:
                if self.qlvalue(state, action) > QV:
                    QV = self.qlvalue(state, action)
                    bestAction = action

        return bestAction

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)  # information
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.mapState(inputs)  # map the states and return

        if self.Learning:
            action = self.choiceAction(self.state) #choice the action but the Q-Learning
        else:
            action = self.choiceRandomAction(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        #get the new state
        self.NEWstate = self.mapState(inputs)  # map the states and return

        #Update the QL with the reward of action and states
        self.QLupdate(reward, action, self.state, self.NEWstate)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs,
                                                                                                    action,
                                                                                                    reward)  # [debug]

        self.storageRewards += reward  # just to pickup some info.


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.000000001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit



if __name__ == '__main__':
    run()
