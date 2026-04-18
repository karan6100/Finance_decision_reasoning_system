from LLMs.llm_factory import LLMFactory
from schema.types import ValidationInput, OverallState


class SummarizerAgent:
    def __init__(self):
        self.llm = LLMFactory.get_llm(task_type='basic')
    
    def run(self, valid_inp: ValidationInput, state: OverallState):
        """
        Summarize and beautify the output from source_agent only if validation passes.
        
        Args:
            valid_inp: ValidationInput containing query, source_agent, candidate_response, etc.
            validation_status: ValidationStatus with decision ("pass" or "fail") and reason
            
        Returns:
            SummarizerOutput with summary and beautified output
        """

        prompt = f"""You are a Summarizer Agent responsible for summarizing validated financial responses.

Original Query: {valid_inp.query}
Source Agent: {valid_inp.source_agent}
Intent: {valid_inp.intent}
Risk Level: {valid_inp.risk}

Validated Response to Summarize and Beautify:
{valid_inp.candidate_response}

Provide a single markdown-formatted response that includes:
1. A concise summary (as a markdown section)
2. Response with proper formatting (headers, bullet points, lists)
3. Professional structure and visual hierarchy
4. Use bold fonts and caution signs wherever there's warning,note or caution.

Use proper markdown syntax with headers, **bold**, bullet points, etc.
"""
        
        response = self.llm.invoke(prompt).content
        return response
        
        # state['summarized_response'] = response
        # return state