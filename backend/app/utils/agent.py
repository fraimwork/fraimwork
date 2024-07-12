import google.generativeai as genai

class Agent:
    def __init__(self, model_name, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.context = []

    def _log_prompt(self, prompt, response):
        self.context.append((prompt, response))

    def _build_prompt(self):
        # Might want to implement more sophisticated context handling here, like summarization or weighting
        return "\n".join([f"{prompt}\nYou: {response}" for prompt, response in self.context])

    def logged_prompt(self, prompt, asker="User"):
        full_prompt = self._build_prompt() + f"\n{asker}: {prompt}"
        response = self.prompt(full_prompt)
        self._log_prompt(f"{asker}: {prompt}", response)
        return response
    
    def prompt(self, prompt):
        return self.model.generate_content(prompt).text
