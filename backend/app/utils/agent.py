import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai.caching import CachedContent
from collections import defaultdict
import asyncio
import time
import networkx as nx

class GenerationConfig:
    def __init__(self, temperature=1.0, top_p=0.95, top_k=64, max_output_tokens=8192, response_mime_type="text/plain"):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.response_mime_type = response_mime_type
    
    def to_dict(self):
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": self.response_mime_type
        }

class SafetySettings:
    def __init__(self, 
                harm_category_hate_speech=HarmBlockThreshold.BLOCK_NONE,
                harm_category_harassment=HarmBlockThreshold.BLOCK_NONE,
                harm_category_sexually_explicit=HarmBlockThreshold.BLOCK_NONE,
                harm_category_dangerous_content=HarmBlockThreshold.BLOCK_NONE):
        self.harm_category_hate_speech = harm_category_hate_speech
        self.harm_category_harassment = harm_category_harassment
        self.harm_category_sexually_explicit = harm_category_sexually_explicit
        self.harm_category_dangerous_content = harm_category_dangerous_content
    
    def to_dict(self):
        return {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: self.harm_category_hate_speech,
            HarmCategory.HARM_CATEGORY_HARASSMENT: self.harm_category_harassment,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: self.harm_category_sexually_explicit,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: self.harm_category_dangerous_content
        }

class Interaction:
    def __init__(self, prompt, response, asker="user", responder="model"):
        self.prompt = prompt
        self.response = response
        self.asker = asker
        self.responder = responder
    
    def to_dict(self):
        return {
            'role': self.asker,
            'parts': [self.prompt],
        }, {
            'role': self.responder,
            'parts': [self.response],
        }

class Agent:
    def __init__(self, model_name, api_key, name, generation_config=GenerationConfig(), system_prompt=None, safety_settings=SafetySettings()):
        genai.configure(api_key=api_key)
        self.api_key = api_key
        self.model_name = model_name
        self.generation_config = generation_config
        self.system_prompt = system_prompt
        self.safety_settings = safety_settings
        self.model = genai.GenerativeModel(model_name, system_instruction=system_prompt, generation_config=generation_config.to_dict(), safety_settings=safety_settings.to_dict())
        self.name = name
    
    def cache_context(context: list[Interaction]):
        contents = [message for interaction in context for message in interaction.to_dict()]
        cached_content = CachedContent.create(
            model_name=self.model_name,
            system_instruction=self.system_prompt,
            contents=contents,
            ttl=datetime.timedelta(minutes=5),
            )
        self.model = genai.GenerativeModel.from_cached_content(cached_content)

    def _log_interaction(self, interaction: Interaction):
        prompt, response = interaction.to_dict()
        time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(f'./logs/{self.name}.log', 'a', encoding='utf-8') as f:
            f.write(f"{prompt['role']}: {prompt['parts'][0]}\n\n{response['role']}: {response['parts'][0]}\n\nLog time: {time_string}\n\n")

    def chat(self, prompt, asker="user", custom_context=[]):
        history = [message for interaction in custom_context for message in interaction.to_dict()]
        session = self.model.start_chat(history=history)
        for _ in range(5):
            try:
                response = session.send_message(prompt).text
                self._log_interaction(Interaction(prompt, response, asker))
                return response
            except Exception as e:
                print(e)
                time.sleep(2)
    
    async def async_chat(self, prompt, asker="user", custom_context: list[Interaction] = []):
        history = [message for interaction in custom_context for message in interaction.to_dict()]
        session = self.model.start_chat(history=history)
        for _ in range(5):
            try:
                response = await session.send_message_async(prompt)
                self._log_interaction(Interaction(prompt, response.text, asker))
                return response.text
            except Exception as e:
                print(e)
                await asyncio.sleep(2)

class Team:
    def __init__(self, *agents: Agent):
        self.agents = {agent.name: agent for agent in agents}
        self.context_threads = defaultdict(list[Interaction])
    
    def chat_with_agent(self, agent_name, message, context_keys=[], save_keys=[], prompt_title=None):
        agent = self.agents[agent_name]
        custom_context = [interaction for key in context_keys for interaction in self.context_threads[key]]
        response = agent.chat(message, custom_context=custom_context)
        user_prompt = message if not prompt_title else prompt_title
        for key in save_keys:
            self.context_threads[key].append(Interaction(user_prompt, response))
        return response

    async def async_chat_with_agent(self, agent_name, message, context_keys=[], save_keys=[], prompt_title=None):
        agent = self.agents[agent_name]
        custom_context = [interaction for key in context_keys for interaction in self.context_threads[key]]
        response = await agent.async_chat(message, custom_context=custom_context)
        user_prompt = message if not prompt_title else prompt_title
        for key in save_keys:
            self.context_threads[key].append(Interaction(user_prompt, response))
        return response
