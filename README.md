
# EquityBot: AI-Powered Equity News Research Tool 

EquityBot is a next-gen research assistant designed for deep financial analysis. Powered by Retrieval-Augmented Generation (RAG) and state-of-the-art generative AI (LLaMA via Groq), it transforms raw news into actionable insights. Users can input URLs of financial articles and ask complex questions, receiving accurate, source-grounded answers in real-time.

Under the hood, EquityBot uses vector embeddings (via HuggingFace’s gte-small), FAISS-based similarity search, and document chunking for scalable retrieval. It's an intelligent blend of LLMs, semantic search, and streamlit interactivity—built for analysts, by analysts.

![](research_tool_sample.jpg)

## Features

- Load URLs or upload text files containing URLs to fetch article content.
- Process article content through LangChain's UnstructuredURL Loader
- Construct an embedding vector using OpenAI's embeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information
- Interact with the LLM's (Chatgpt) by inputting queries and receiving answers along with source URLs.


## Installation

1.Clone this repository to your local machine using:

```bash
  git clone https://github.com/rudhreesh/https://github.com/rudhreesh/EquityBot.git
```
2. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
3.Set up your Groq API key by creating a .env file in the project root and adding your API

```bash
  GROQ_API_KEY=your_api_key_here
```
## Usage/Examples

1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser.

- On the sidebar, you can input URLs directly.

- Initiate the data loading and processing by clicking "Process URLs."

- Observe the system as it performs text splitting, generates embedding vectors, and efficiently indexes them using FAISS.

- The embeddings will be stored and indexed using FAISS, enhancing retrieval speed.

- The FAISS index will be saved in a local file path in pickle format for future use.
- One can now ask a question and get the answer based on those news articles
- In video tutorial, we used following news articles
  - https://www.indiatoday.in/business/market/story/sensex-down-nifty-falls-market-crash-kashmir-terror-attack-pakistan-nse-bse-dalal-street-red-2714878-2025-04-25
  - https://m.economictimes.com/markets/stocks/stock-watch/stock-market-update-stocks-that-hit-52-week-lows-on-nse-in-todays-trade/amp_articleshow/120583747.cms
  - https://www.etnownews.com/markets/indian-stock-market-tomorrow-april-24-donald-trump-may-slash-china-tariffs-dow-jones-nasdaq-open-with-gains-how-will-it-impact-bse-nse-article-151481380

## Project Structure

- main.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.
- faiss_store_huggingface.pkl: A pickle file to store the FAISS index.
- .env: Configuration file for storing your OpenAI API key.
