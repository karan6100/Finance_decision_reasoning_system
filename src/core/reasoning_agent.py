from LLMs.llm_factory import LLMFactory
from schema.types import ReasoningInput, ReasonOutput


class ReasonAgent:

    def __init__(self):
        self.llm = LLMFactory.get_llm(task_type='reasoning')

    def run(self, inp: ReasoningInput)-> ReasonOutput:

        prompt = f""" You are a Finance Reasoning Agent.
                    The system has already classified:
                    - User intent (education, analysis, recommendation, factual)
                    - Risk level (low, medium, high)

                    You must treat these classifications as authoritative inputs.

                    Your task is to:
                    - Perform structured financial reasoning within the given intent and risk constraints
                    - Decide whether external tools are required
                    - Produce a clear, balanced, and compliant response

                    Reasoning Process:
                    1. Reason:
                    - Interpret the query using the provided intent and risk labels
                    - Identify missing assumptions or uncertainty
                    - Decide whether reasoning alone is sufficient

                    2. Act (only if required):
                    - Use tools for calculations, retrieval, or verification
                    - Never fabricate data

                    3. Observe:
                    - Validate tool outputs
                    - Check for inconsistencies or gaps

                    4. Final Answer:
                    - Explain reasoning clearly
                    - State assumptions and limitations
                    - Avoid guarantees or personalized advice unless explicitly permitted

                    Below is the user input and its classifications:\n
                    input: {inp.query} \n
                    intent: {inp.intent} \n
                    risk level: {inp.risk} \n
                    """

        # Response
        response = self.llm.invoke(prompt).content
        return ReasonOutput(
            reasoning= response
        )
