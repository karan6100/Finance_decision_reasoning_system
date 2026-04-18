from LLMs.llm_factory import LLMFactory
from schema.types import BasicInput, BasicOutput

class BasicAgent:

    def __init__(self):
        self.llm = LLMFactory.get_llm(task_type='basic')

    def run(self, inp: BasicInput) -> BasicOutput:

        prompt= f"""Provide a factual, evidence-based response to the user's input. 
        Ensure all information is accurate, objective, and educational in nature. 
        Include references to credible educational articles, research papers, or authoritative sources where relevant.
        Avoid speculation, opinions, or unverified claims. Focus on clarity, correctness, and learning value.
        Below is the user input and its classifications:\n
        input: {inp.query}
                """
        
        # Response
        response = self.llm.invoke(prompt).content
        return BasicOutput(
            output= response
        )
