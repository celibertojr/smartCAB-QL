import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple
import time

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""



    def __init__(self, env):
        super(LearningAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'blue'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.alpha = 0.2#alpha value
        self.gamma = 0.9 #gama value
        self.state = None
        self.action = None
        self.OLDstate = None
        self.NEWstate = None
        self.agent_reward = None
        self.storageRewards = 0  # just for testing
        self.countFail = 0 #just for test the number of Fails
        self.QL = dict()  # storage the state and action here. (https://docs.python.org/2/library/stdtypes.html)
        self.exploration_rate = 0.1  # percentage of randomness (10%)
        self.Learning = True # choice Learning (default QL) ou random
        self.Sarsa = True # choice QL or S.A.R.S.A
        self.addStorage=0 #debug
        self.Allsteps =0 #debug
        self.addsteps = 0#debug
        self.addAllsteps = 0 #debug
        self.penalties = [] #debug
        self.penalty = 0 #debug
        self.Allpenalties=0 #debug



    def reset(self, destination=None):
        """
        initializes the environment variables
        input: env, initializes the environment variable for sensing

        Responsible for initializing the color of the car, and the route.
        for the learning start the reward with 0 and variables station and action
        empty.
        """
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.action = None
        self.storageRewards = 0
        self.penalties.append(self.penalty)
        self.Allpenalties += self.penalty
        self.penalty = 0





    def qlvalue(self, state, action):  # return the Q value give state and action
        """
        in=> state and action
        out<= Q value for the state and action
        returns 0 if the value is not present in the dictionary.
        """
        return self.QL.get((state, action), 20)

    def mapActions(self, state):  # return info about legal action from state
        return ['forward', 'left', 'right', None]

    def mapState(self, state): #return  info about legal actual state
        State = namedtuple("State", ["light", "next_waypoint"])
        return State(light=state['light'],
                     next_waypoint=self.planner.next_waypoint())

    def best_qvalue(self, state):
        """
            Returns the value of best action from the state
        """
        Actions = self.mapActions(state)
        qv = - 999999999
        for action in Actions:
            if self.qlvalue(state, action) > qv:
                qv = self.qlvalue(state, action)
        return qv

    def choiceAction(self, state):
        """
        Choice the best action to take.

        Here is the heart of the learning. From all the action that
        the agent is allowed to do, the algorithm return the best to
        do. But with 10% of randomness to avoid to be stuck in a local minimum

        """

        Actions = self.mapActions(state)
        bestAction = None
        QV = - 999999999

        rvalue = random.random()
        if rvalue < self.exploration_rate:  # do a random action
            bestAction = random.choice(Actions)
        else:
            for action in Actions:
                if self.qlvalue(state, action) > QV:
                    QV = self.qlvalue(state, action)
                    bestAction = action

        return bestAction

    def QLupdate(self, reward, action, state, nextState):
        """
            Update the QLearning algorithm.
            Q(s,a)=Q(s,a)+alpha*(reward + gama(maxQ(s',a') - Q(s,a))

            """
        self.QL[(state, action)] = self.qlvalue(state, action) + self.alpha * (
            reward + (self.gamma * self.best_qvalue(nextState)) - self.qlvalue(state, action))


    def Sarsaupdate(self, reward, action, state, nextState):
        """
            Update the SARSA algorithm.
            Q(s,a)=Q(s,a)+alpha*(reward + gama((s',a') - Q(s,a))

            """
        action2= self.choiceAction(nextState)
        self.QL[(state, action)] = self.qlvalue(state, action) + self.alpha * (
            reward + (self.gamma * self.qlvalue(nextState, action2)) - self.qlvalue(state, action))

    def choiceRandomAction(self, state):
        #choice random action

        Actions = self.mapActions(state)
        bestAction = random.choice(Actions)
        return bestAction



    def update(self, t):
        """
            Main update method that is responsible for updating the agent action.
            """
        # Gather inputs


        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)  # information
        deadline = self.env.get_deadline(self)


        # TODO: Update state
        self.state = self.mapState(inputs)  # map the states and return

        if self.Learning:
            action = self.choiceAction(self.state) #choice the action from the Q-table
        else:
            action = self.choiceRandomAction(self.state) # choice an random action

        # Execute action and get reward
        reward = self.env.act(self, action)

        # calculate penalties
        if reward < 0:
            self.penalty += 1


        # TODO: Learn policy based on state, action, reward

        #get the new state
        self.NEWstate = self.mapState(inputs)  # map the states and return

        #Update the QL or SARSA with the reward of action and states
        if self.Sarsa:
            self.Sarsaupdate(reward, action, self.state, self.NEWstate)
        else:
            self.QLupdate(reward, action, self.state, self.NEWstate)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs,
                                                                                                    action,
                                                                                                    reward)  # [debug]

        self.storageRewards += reward  # just to pickup some info of reward

        self.addStorage += reward
        self.addsteps+=(self.Allsteps-deadline)
        self.addAllsteps +=self.Allsteps


        print "Values: Storage = {}, AllSteps={}. SmartSteps = {}, Penalties = {}".format(self.addStorage, self.addAllsteps,
                                                                                          self.addsteps,
                                                                                          self.Allpenalties)  # [debug]
        #print self.addStorage # [debug]
        #print self.addAllsteps # [debug]
        #print self.addsteps # [debug]
        #print self.Allpenalties # [debug]





def run():
    """Run the agent for a finite number of trials."""
    start = time.time()
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.00001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    end = time.time()
    print"Time (s) = {}".format((end - start))
    print(a.penalties)

if __name__ == '__main__':
    run()
