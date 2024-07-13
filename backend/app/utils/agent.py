import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from collections import defaultdict
import time

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
    def __init__(self, prompt, response, asker="user"):
        self.prompt = prompt
        self.response = response
        self.asker = asker
    
    def to_dict(self):
        return {
            'role': self.asker,
            'parts': [self.prompt],
        }, {
            'role': 'model',
            'parts': [self.response],
        }

class Agent:
    def __init__(self, model_name, api_key, name="Agent", generation_config=GenerationConfig(), system_prompt=None, safety_settings=SafetySettings()):
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(model_name, system_instruction=system_prompt, generation_config=generation_config.to_dict(), safety_settings=safety_settings.to_dict())
        self.keyed_interactions = defaultdict(list[Interaction])
        self.name = name

    def _log_interaction(self, interaction: Interaction):
        prompt, response = interaction.to_dict()
        time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(f'./logs/{self.name}.log', 'a', encoding='utf-8') as f:
            f.write(f"{prompt['role']}: {prompt['parts'][0]}\n\n{response['role']}: {response['parts'][0]}\n\nLog time: {time_string}\n\n")

    def _build_context(self, key="all", context=None):
        if not key: return []
        elif context: return [message for interaction in context for message in interaction.to_dict()]
        return [message for interaction in self.keyed_interactions[key] for message in interaction.to_dict()]
    
    def add_to_context(self, prompt, response, asker, key="all"):
        interaction = Interaction(prompt, response, asker)
        self.keyed_interactions["all"].append(interaction)
        if key != "all": self.keyed_interactions[key].append(interaction)
        self._log_interaction(interaction)
    
    def chat(self, prompt, context_key=None, asker="user", save_context="all", custom_context=None):
        history = self._build_context(context_key, custom_context)
        session = self.model.start_chat(history=history)
        response = session.send_message(prompt).text
        self.add_to_context(prompt, response, asker, save_context)
        return response
