# Tom GPT

This repository is for learning generative AI.

## Table of Contents

- Project Structure
- get-started
- Installation

## Project Structure

tom-gpt/  
├── get-started/  
│   └── app.py  
│   └── requirements.txt
├── with-rag/  
│   └── app.py  
│   └── requirements.txt
├── with-react-agent/  
│   └── app.py  
│   └── requirements.txt
├── .gitignore  
└── README.md  

## with-react-agent

This single page app is for demostrating the following tech stack or ideas.

- LLM Cohere
  - Model Command R+
  - Role, such as ["system", "user", "assistant"]
- Prompt Engineering
  - ReAct Prompting
- Framework LangChain
  - LangChain Agent, such as AgentExecutor, Tool, create_react_agent
  - LangChain Chain, such as LLMMathChain
  
## with-rag or get-started

This single page app is for demostrating how to use RAG, on top of with react-agent.

- RAG
  - Embeddings CohereEmbeddings
  - Vector Store PineconeVectorStore
  - LangChain Chain RetrievalQA
  
Before that, related documents for RAG is required to store in object store (e.g. AWS S3) and embedd in vector store (e.g. Pinecone)

## Installation

This app is hosted by Streamlit Cloud.

[Visit Streamlit](https://streamlit.io/)  
[Viist Demo app](https://tom-gpt.streamlit.app/)
