#Creating a new function
#It gets a name and returns its Linkedin URL
import os
from langchain import PromptTemplate

#Getting Google ChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

#Getting Langchain Agents
#from langchain.agents import initialize_agent, Tool, AgentType, create_react_agent

from langchain_core.tools import Tool

#Getting profile_url function from tools.py file
from tools.tools import get_profile_url

from langchain import hub
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)

def lookup(name: str) -> str:
    google_llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0,
                                        google_api_key=os.environ["GOOGLE_API_KEY"])
    template = """Given the full name {name_of_person} , give me its full linkedin URL. 
    I want a valid and workable Linkedin URL.
    Your output should only contain a Linkedin URL"""

    prompt_template = PromptTemplate(template=template, input_variables=['name_of_person'])

    tools_for_agent = [Tool(name = "Crawl Google 4 linkedin profile page", func = get_profile_url,
                            description="Useful for when you need to get linkedin page URL")]
    """
    agent = initialize_agent(tools =tools_for_agent, llm =google_llm,
                             agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                             verbose =True) """

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=google_llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linkedin_profile_url = result["output"]

    return linkedin_profile_url