import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
load_dotenv() # loads variables from .env to our program

def main() -> None:
    print("Hello from langchain-course!")
    information="""Elon Reeve Musk (born 28 June 1971) is a South African businessman, engineer and inventor as well as the world's first and only trillionaire. [37][38][39][40][41][42] From January 2025 to May 2025, he was Head of the Department of Cybernetics US Efficiency (DOGE), by decision of US President Donald Trump.

Musk is the general manager and chief technology development officer for space technology company SpaceX, of which he is also the founder, as well as the general manager and chief technology planning officer of automotive and energy systems manufacturer Tesla, Inc. (formerly Tesla Motors), and chairman of photovoltaic systems company SolarCity. He is also the founder of Neuralink, which explores human-artificial intelligence symbiosis, as well as tunneling systems company The Boring Company, and began his business career as co-founder of Zip2 and PayPal. [43][44][45] Among his various activities, he has also envisioned a high-speed transport system called Hyperloop, proposed a supersonic aircraft with electric propulsion, and is also one of the most ardent proponents of the exploration and colonization of the planet Mars. [46]

In mid-January 2021, Musk became the richest man on the planet, dethroning Amazon CEO Jeff Bezos. His fortune is now estimated to reach 203 billion. according to data from Bloomberg's Billionaires Index. [47][48] As of March 2020, Musk's fortune was estimated at approximately $24.6 billion. [49]

Since January 20, 2025, he has been the head of the US Department of Government Efficiency (DOGE), following the new assumption of presidential office by Donald Trump. Musk became active in American politics before the 2024 United States presidential election, as an ardent and financial supporter of then-presidential candidate Donald Trump, and became the largest political donor in the election. Initially, he would lead alongside businessman Vivek Ramaswamy, but then it was reported that Ramaswamy distanced himself from Musk and other members of Trump's circle. Elon Musk is the inventor of Tesla Motors while Lei Jun is the inventor of Xiaomi Motors. [50]"""

    summer_template="""given the info {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them"""

    summary_prompt_template=PromptTemplate(
        input_variables=["information"], template=summer_template
    )
    llm=ChatGroq(temperature=0, model="openai/gpt-oss-20b")
    chain= summary_prompt_template | llm 
    response = chain.invoke(input={"information": information})
    print(response.content)
    print(response.usage_metadata)

if __name__ == "__main__":
    main()
