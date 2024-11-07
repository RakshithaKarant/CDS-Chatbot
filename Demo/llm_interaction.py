import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from CustomLLM import CustomerSupportTransformersLLM as my_llm
import os
class llm_Demo:
    def __init__(self):
        load_dotenv()
        model_name = os.environ['model_path']  # Replace with the path to your local model


        # Initialize custom LLM with locally loaded model
        self.custom_llm = my_llm(model_name)

        # Initialize conversation memory
        self.memory = ConversationBufferMemory()

        # Create a ConversationChain with the custom LLM and memory
        self.chat_model = ConversationChain(
            llm=self.custom_llm,
            memory=self.memory,
            verbose=True  # Set to True if you want to see intermediate steps
        )

    def _set_template(self, query: str) -> str:
        if re.search(r"\bhaven't received\b|\blate\b", query, re.IGNORECASE):
            template_s = """
                You are a customer service agent using empathy.
                The customer said: "{query}"
                Your response: "I understand how frustrating this might be. Let me help you as quickly as possible."
            """
        elif re.search(r"\bbroken\b|\bnot working\b", query, re.IGNORECASE):
            template_s = """
                You are a customer service agent helping to solve a problem.
                The customer said: "{query}"
                Your response: "I’m sorry you’re experiencing this issue. Let me guide you step-by-step to resolve it."
            """
        else:
            template_s = """
                You are a customer service agent handling escalations.
                The customer said: "{query}"
                Your response: "It looks like I’m unable to assist you fully at the moment. I’m escalating your issue to our support team."
            """
        prompt_template = ChatPromptTemplate.from_template(template_s)
        return prompt_template.format(query=query)

    def initiate_chat(self, customer_query: str) -> None:
        # Format the query with the appropriate response template
        formatted_query = self._set_template(customer_query)

        # Use the ConversationChain to get the bot's response
        response = self.chat_model.predict(input=formatted_query)

        print(f"Bot Response: {response}")

if __name__ == "__main__":
    queries = [
        "Hi, I am John. I haven’t received my order yet, and it’s been delayed for days.",
        "My product is broken, and I can’t get it to work.",
        "I want to cancel my order."
    ]

    llm_demo = llm_Demo()

    for i, query in enumerate(queries, 1):
        print(f"\nResponse for Query {i}:")
        llm_demo.initiate_chat(query)
