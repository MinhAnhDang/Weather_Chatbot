import streamlit as st
import json
from streamlit_chat import message
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama

def submit():
    st.session_state.query = st.session_state.input
    st.session_state.input = ""

# Define tool to get weather information
@tool    
def get_weather(location:str):
    "Get weather information of any location"
    response = {"location": location,
                "temperature": "22", 
                "Humidity": "93%",
                "Air Quality": "115",
                "Visibility": "10km",
                "Wind": "10km/h",
                "weather": "Cloudy"}
    return json.dumps(response)   
 
#Initialize llm model
llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

model = ChatHuggingFace(llm=llm, verbose=False)

#Define tool metadata
tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather of certain place",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location of place to get information",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        ]

#equip llm with get_weather information tool
model = model.bind_tools(tools=tools)

st.subheader("Weather Chatbot")
if "query" not in st.session_state:
    st.session_state.query = ""
    
if 'responses' not in st.session_state:
    # Greet user and ask for user's name
    st.session_state['responses'] = ["Hello. I'm Cake, an weather assistant. What's your name?"]
    
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory=[]
    st.session_state.buffer_memory.append(SystemMessage(content="You are a helpful assistant"))
    st.session_state.buffer_memory.append(AIMessage(content="Hello. I'm Cake. What's your name?"))


# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

tool_lists = {"get_weather": get_weather}
with textcontainer:
    st.text_input("Query: ", key="input", on_change=submit)
    query = st.session_state.query

    if query :
        st.session_state.buffer_memory.append(HumanMessage(content=query))
        with st.spinner("typing..."):
            response = model.invoke(st.session_state.buffer_memory)
            # Check if model call tool
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    args = tool_call['args']
                    # Run the tool to get information
                    response = tool_lists[tool_name].invoke(tool_call)
                    print(response)
                    # Add tool response to memory
                    st.session_state.buffer_memory.append(ToolMessage(content=response, tool_call_id=tool_call['id'], name=tool_name))
                # print(st.session_state.buffer_memory)
                # Run llm again with tool response to get final response
                response = model.invoke(st.session_state.buffer_memory)
                
        # Add request and response to session_state in order to print out in mornitor      
        st.session_state.requests.append(query)
        st.session_state.responses.append(response.content)
        st.session_state.buffer_memory.append(AIMessage(content=response.content))

# Print out request and response
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

