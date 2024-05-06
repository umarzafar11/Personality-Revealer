# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import langchain
# import pandas
# import numpy
# import flask
# from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
import warnings
from langchain._api import LangChainDeprecationWarning

from output_parser import person_intel_parser, PersonIntel

warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# For multi-chains:
from langchain.chains import SimpleSequentialChain

from third_parties.linkedin import scrape_linkedin_profile

#Google API Key:
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

#Getting Twitter.py file instance
from third_parties.twitter import scrape_user_tweets



#Encapsulate everything inside a FUnction:
def icebreak(name: str) -> tuple[PersonIntel, str]:
    linkedin_profile_url = linkedin_lookup_agent(name=name)

    summary_template = """I want to get the following information about the personality {information} :
                           1) A short Summary
                           2) Two interesting Facts about them
                           3) A topic that may interest them
                           3) Creative Icebreaker to open a conversation with them
                                    \n{format_instructions}
                           """

    summary_prompt_template = PromptTemplate(input_variables=['information'],
                                             template=summary_template,
                                             partial_variables = {"format_instructions":person_intel_parser.get_format_instructions()},
                                             )

    # llm = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo')
    google_llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)

    chain = LLMChain(llm=google_llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(
        # For importing pre-saved Json in my Github Gist
        # linkedin_profile_url="https://gist.githubusercontent.com/umarzafar11/923d9d36dc1a06670d3907e05f3c7d04/raw/9f4713c97b9390a88ec0c0406a021370c2a3adf8/umar_zafar.json"
        linkedin_profile_url=linkedin_profile_url
    )

    print('Linkedin profile url: '+linkedin_profile_url)
    result = chain.run(information=linkedin_data)
    print(result)

    return person_intel_parser.parse(result), linkedin_data.get('profile_pic_url')


if __name__ == '__main__':
    print('Pycharm')

    #Calling IceBreak
    result = icebreak(name='Reed Hastings')
    print('Final Results')
    print(result)


    """
    linkedin_profile_url = linkedin_lookup_agent(name='Umar Zafar Kaggle')
    #linkedin_profile_url = 'https://www.linkedin.com/in/melanieperkins/'

    summary_template = I want to get the following information about the personality {information} :
                               1) A short Summary
                               2) Two interesting Facts about them
                               3) A topic that may interest them
                               3) Creative Icebreaker to open a conversation with them
                               

    summary_prompt_template = PromptTemplate(input_variables=['information'],
                                             template=summary_template,
                                             )

    # llm = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo')
    google_llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)

    chain = LLMChain(llm=google_llm, prompt=summary_prompt_template)
    print('Linkedin profile url: ' + linkedin_profile_url)
    linkedin_data = scrape_linkedin_profile(
        # For importing pre-saved Json in my Github Gist
        # linkedin_profile_url="https://gist.githubusercontent.com/umarzafar11/923d9d36dc1a06670d3907e05f3c7d04/raw/9f4713c97b9390a88ec0c0406a021370c2a3adf8/umar_zafar.json"
        linkedin_profile_url=linkedin_profile_url
    )

    print(linkedin_data)

    result = chain.run(information=linkedin_data)
    print(result)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
"""

#Get Tweets
#print(scrape_user_tweets(username='Donald Trump'))