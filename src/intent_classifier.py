from openai import OpenAI


class IntentClassifier:
    """Agent responsible for classifying user intent using binary classification"""
    
    def __init__(self, client: OpenAI, model: str, system_prompt: str = None):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
    
    def classify(self, user_input: str) -> str:
        """
        Classify user input and return the binary result.
        Returns '0' or '1' based on the classification.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=1,
            temperature=0,
            logit_bias={
                "15": 100,  # Token for "0"
                "16": 100   # Token for "1"
            }
        )
        return response.choices[0].message.content.strip()
    
    def is_positive(self, user_input: str) -> bool:
        """
        Classify and return True if result is '1' (positive case).
        Useful for yes/no, exit/continue, positive/negative classifications.
        """
        return self.classify(user_input) == "1"
