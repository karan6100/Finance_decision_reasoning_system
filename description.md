What the system will do?
- The system first evaluates the user query using a Decision Gate, classifying it by intent (factual, analytical, opinion) and risk intensity.
- Based on this classification, an Orchestrator selects an appropriate execution strategy, such as constrained reasoning, retrieval-based analysis, or refusal pathways.
- A Reasoning Agent or API-backed module performs structured analysis or information retrieval without directly producing user-facing output.
- An Answer Generator converts internal reasoning into a user-facing response while respecting tone, confidence bounds, and safety constraints.

- A Review (Validation) Agent evaluates the generated output across three dimensions:
    1. factual consistency
    2. reasoning soundness
    3. risk compliance
- Based on this evaluation, the system takes one of four actions:
    1. Valid → forward the response to a summarization step and return it to the user
    2. Insufficient → re-enter the reasoning flow within bounded retries or confidence thresholds
    3. Underspecified → request targeted clarification from the user
    4. Unsafe → refuse to answer with a transparent explanation

What the system won't do?
- It will not provide personalized or prescriptive financial advice without essential user context such as investment horizon, risk tolerance, or financial objectives.
- It will not make speculative guarantees or predictions in high-uncertainty scenarios (e.g., short-term market movements).
- It will not answer high-risk opinion-based queries when confidence or evidence thresholds are not met.

When it refuses?
- The system refuses when a query is classified as high-risk and opinion-based, and the response would rely on uncertain, subjective, or speculative factors (e.g., market timing, price guarantees).
- Refusal is accompanied by a clear explanation of why the request cannot be answered safely, and where possible, the system redirects the user toward safer, educational alternatives.

When it asks for clarification?
- The system requests clarification when the validation step determines that essential user-provided information is missing, and that the insufficiency cannot be resolved through retrieval or reasoning alone.03
- Clarification requests are targeted and derived from failed validation criteria, not generic follow-up questions.