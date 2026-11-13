from LLMs.llm_factory import LLMFactory
from schema.types import ReasoningInput, ReasonOutput, OverallState


class ReasonAgent:

    def __init__(self):
        self.llm = LLMFactory.get_llm(task_type='reasoning')

    def run(self, inp: ReasoningInput, state: OverallState)-> ReasonOutput:

        # Feedback from previous failed attempt
        feedback = ""
        reviews = state.get("valid_reviews") or {}
        if state.get("valid_status") == "fail" and reviews.get("reason"):
            if not reviews.get("parse_error", False):
                # prev = (state.get("reasons") or [""])[-1]
                prev = state.get("reasoning_response", "")
                feedback = f"""
                        --- RETRY: PREVIOUS ATTEMPT WAS REJECTED ---
                        Your previous answer:
                        <previous_answer>
                        {prev}
                        </previous_answer>

                        Validator's reason for rejection:
                        <validator_feedback>
                        {reviews["reason"]}
                        </validator_feedback>

                        Write a new answer that fixes every issue the validator raised.
                        Do not repeat the previous answer verbatim.
                        """


        prompt = f""" You are a Finance Reasoning Agent.
                    The system has already classified:
                    - User intent (education, analysis, recommendation, factual)
                    - Risk level (low, medium, high)

                    You must treat these classifications as authoritative inputs.

                    Your task is to:
                    - Perform structured financial reasoning within the given intent and risk constraints
                    - Produce a clear, balanced, and compliant response

                    Constraints:
                    - You have NO tools, NO internet access, and NO live data. Do not call or simulate any tool.
                    - Do not cite specific studies, articles, or statistics unless you are certain they exist;
                    prefer general, well-established principles.

                    Reasoning Process:
                    1. Understand:
                    - Interpret the query using the provided intent and risk labels
                    - Identify missing assumptions or uncertainty
                    - If the question depends on current data (prices, rates, returns), say which data the
                    user would need to check, and answer the parts that do not depend on it

                    2. Analyze:
                    - Reason from general financial knowledge; compare options and trade-offs
                    - If you mention a specific figure, mark it as approximate and possibly outdated

                    3. Self-check before answering:
                    - No guarantees, certainty about returns, or invented figures/sources
                    - Assumptions and limitations are stated
                    - No contradictions or gaps in the reasoning

                    4. Final Answer:
                    - Explain reasoning clearly
                    - State assumptions and limitations
                    - Do not give guarantees or personalized advice


                    Below is the user input and its classifications:\n
                    input: {inp.query} \n
                    intent: {inp.intent} \n
                    risk level: {inp.risk} \n
                    {feedback}
                    
                    """

        # Response
        response = self.llm.invoke(prompt).content
        return ReasonOutput(
            reasoning= response
        )
