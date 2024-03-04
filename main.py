from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from save_engine import save_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI


# load the environment variables
load_dotenv()

# specify the path to the data file
diseaseData_path = os.path.join('data', 'Disease.csv')
disease_df = pd.read_csv(diseaseData_path)

# print(disease_df.head())

# create a query engine
diseases_query_engine = PandasQueryEngine(df=disease_df, verbose=True, instruction_str=instruction_str)

# update the prompt
diseases_query_engine.update_prompts({
    "pandas_prompt": new_prompt
})

# test by giving a query
# diseases_query_engine.query("What is the most common disease in the dataset?")

# Specify the tools we have access to
tools = [
    save_engine,
    QueryEngineTool(query_engine=diseases_query_engine, metadata=ToolMetadata(
        name="Disease_Data",
        description="This helps identify the disease based on the symptoms and provide first aid instructions",
    ))
]

# Create the agent
llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while(prompt := input("Enter a prompt (or 'q' to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)