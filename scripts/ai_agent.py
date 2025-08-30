import groq
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

# Load API key from environment
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing. Set it in .env or as an environment variable.")

# Define AI Model
llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant", temperature=0)

class CleaningState(BaseModel):
    """State schema defining input and output for the LangGraph agent."""
    input_text: str
    structured_response: str = ""

class AIAgent:
    def __init__(self):
        self.graph = self.create_graph()

    def create_graph(self):
        """Creates and returns a LangGraph agent graph with state management."""
        graph = StateGraph(CleaningState)

        # ✅ FIX: Ensure agent outputs structured response
        def agent_logic(state: CleaningState) -> CleaningState:
            """Processes input and returns a structured response."""
            response = llm.invoke(state.input_text)
            # Ensure structured_response is a string
            content = getattr(response, "content", str(response))
            return CleaningState(input_text=state.input_text, structured_response=content)

        graph.add_node("cleaning_agent", agent_logic)
        graph.add_edge("cleaning_agent", END)
        graph.set_entry_point("cleaning_agent")
        return graph.compile()

    def process_data(self, df, batch_size=20):
        """Processes data in batches to avoid OpenAI's token limit."""
        cleaned_responses = []

        for i in range(0, len(df), batch_size):
            df_batch = df.iloc[i:i + batch_size]  # ✅ Process 20 rows at a time

            prompt = f"""
            You are an AI Data Cleaning Agent. Analyze the dataset:

            {df_batch.to_string()}

            Identify missing values, choose the best imputation strategy (mean, mode, median), 
            remove duplicates, and format text correctly.

            Return the cleaned data as structured text.
            """

            state = CleaningState(input_text=prompt, structured_response="")
            response = self.graph.invoke(state)

            if isinstance(response, dict):
                response = CleaningState(**response)

            cleaned_responses.append(response.structured_response)  # ✅ Store results

        return "\n".join(cleaned_responses)  # ✅ Combine all cleaned results
