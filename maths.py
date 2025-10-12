import validators,streamlit  as st 
from langchain import PromptTemplate
from langchain_groq import ChatGroq 
from langchain.chains import LLMChain,LLMMathChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool,initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
import os

##set up the streamlit app
st.set_page_config(page_title="Text to Math problem solver and Data Search Assistant",page_icon="🦜")
st.title("🦜 Text to Math problem solver and Data Search Assistant using Google gemma 2")
groq_api_key=st.sidebar.text_input(label="groq_api_key",type="password")
if not groq_api_key:
    st.info("Please enter your groq_api_key to use the app")
    st.stop()

llm=ChatGroq(groq_api_key=groq_api_key,model="llama-3.1-8b-instant")

##initialize the tool
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="wikipedia",
    func=wikipedia_wrapper.run,
    description="useful for when you need to look up a topic or get specific information about people,places,or things"
)

##inirialize the math tool or math chain
## simplified calculator without LLMMathChain
def simple_calculator(question):
    response = llm.invoke(f"Calculate the result for: {question}. Only return the numeric answer, no explanation.")
    return response.content.strip()

calculator = Tool(
    name="calculator",
    func=simple_calculator,
    description="useful for solving mathematical problems quickly and returning only the numeric result"
)

prompt="""you are an expert data analyst and python programmer. You can perform mathematical calculations and data analysis. 
Question:{question}
answer:"""

prompt_template=PromptTemplate(template=prompt,input_variables=["question"])

#let's create the chain which will combine all the tools which we have created above
#1) create llm chain
chain=LLMChain(llm=llm,prompt=prompt_template)

##add reasoning
reasoning_tool=Tool(
    name="reasoning",  
    func=chain.run,
    description="useful for when you need to think step by step and break down a problem"
)

##initilaize the agent
assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,   #it get print in console thats why keep it false
    handle_parsing_errors=True
)

##this is what we do commonly when we use session state
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'] )   

#let's start the interaction
question=st.text_area("enter your question:")
##make sure to install wikipedia

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response"):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])
            #response = assistant_agent.run(question, callbacks=[st_cb])
        
            st.session_state.messages.append({"role":"assistant","content":response})
            print(response)
            st.write("### response")
            st.success(response)
    else:
        st.warning("please enter your question")