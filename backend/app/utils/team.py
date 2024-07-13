from utils.agent import Agent

class Team:
    def __init__(self, agents: list[Agent]):
        self.agents = agents

    def ping_all_agents(self, message, messenger="global", context_key=None):
        for agent in self.agents:
            if agent == messenger: yield None; continue
            yield agent.chat(message, asker=messenger, context_key=context_key)