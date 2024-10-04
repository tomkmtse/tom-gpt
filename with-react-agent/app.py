import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.chains import LLMMathChain
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.runnables import RunnableConfig
from langchain_cohere import ChatCohere

SYSTEM_MSG = "In this version, I would think, act (search or calculate) and observe step by step until coming up with a final answer. Anything I can help?"
TEMPLATE = '''
Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
Thought:{agent_scratchpad}
'''

avatar = lambda role: "🤳" if role == 'user' else "🤖"

# LLM
llm = ChatCohere(cohere_api_key=st.secrets["cohere_api_key"], streaming=True)

# Tools
search = DuckDuckGoSearchAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    )
]

# Prompt
prompt = PromptTemplate.from_template(template=TEMPLATE)

# Agent
react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True)

# View
st.set_page_config(
    page_title="Tom GPT", page_icon="🤖", layout="wide"
)

"# 🤖 Tom GPT"

## Anonymous system container
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": SYSTEM_MSG}]
    
for message in st.session_state.chat_history:
    st.chat_message(name=message["role"], avatar=avatar(message["role"])).markdown(body=message["content"])

if user_input := st.chat_input(placeholder="Your question/ instruction"):
    ## Anonymous user container
    st.chat_message(name="user", avatar=avatar("user")).markdown(body=user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    ## Assistant container
    assistant_container = st.chat_message(name="assistant", avatar=avatar("assistant"))
    st_callback = StreamlitCallbackHandler(parent_container=assistant_container)
    runnable_config = RunnableConfig(callbacks=[st_callback])
    
    answer = agent_executor.invoke(input={"input": user_input, "chat_history": st.session_state.chat_history}, config=runnable_config)
    assistant_container.markdown(body=answer["output"])
    st.session_state.chat_history.append({"role": "assistant", "content": answer["output"]})