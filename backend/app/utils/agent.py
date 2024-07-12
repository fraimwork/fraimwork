import google.generativeai as genai

class Agent:
    def __init__(self, model_name, api_key, name="Agent"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.context = []
        self.logs = []
        self.name = name

    def _log_prompt(self, prompt, response):
        self.context.append((prompt, response))

    def _build_prompt(self):
        # Might want to implement more sophisticated context handling here, like summarization or weighting
        return "\n".join([f"{prompt}\nYou: {response}" for prompt, response in self.context])

    def logged_prompt(self, prompt, asker="User"):
        full_prompt = self._build_prompt() + f"\n{asker}: {prompt}\nYou: "
        response = self.prompt(full_prompt, asker=asker)
        self._log_prompt(f"{asker}: {prompt}", response)
        return response
    
    def prompt(self, prompt, asker="User"):
        response = self.model.generate_content(prompt).text
        self.logs.append((f"{asker}: {prompt}", response))
        return response
    
    def log_conversation(self, path):
        with open(path, "w") as f:
            for prompt, response in self.logs:
                f.write(f"{prompt}\nYou: {response}\n")
