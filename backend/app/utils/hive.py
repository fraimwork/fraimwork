from utils.agent import Agent

class Hive:
    def __init__(self, agents: list[Agent]):
        self.agents = agents

    def ping_all_agents(self, message, messenger="Global"):
        for agent in self.agents:
            if agent == messenger:
                continue
            agent.logged_prompt(message)

    def log_all_agents(self):
        for agent in self.agents:
            agent.log_conversation()