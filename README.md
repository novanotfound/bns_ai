✅ README.md for your project

Create a new file called README.md in your project root and paste this:

# ⚖️ BNS Legal AI Assistant

An AI-powered legal assistant for the **Bharatiya Nyaya Sanhita (BNS)** built using a structured knowledge graph + large language model.  
The system answers legal questions with **grounded citations**, **intuitive explanations**, and **anti-hallucination safeguards**.

---

## 🚀 Key Features

- ✅ **BNS Knowledge Graph**  
  Structured graph containing legal sections, offences, penalties, and relationships.

- ✅ **Graph-Grounded LLM Responses**  
  Responses are generated using retrieved legal nodes as *ground truth*, reducing hallucination.

- ✅ **Two-Stage Reasoning Pipeline**
  1. Free-form LLM reasoning  
  2. Legal refinement and validation using the graph

- ✅ **Critic Verification System**  
  Every answer is reviewed by an internal verifier:
  - `PASS`  
  - `PASS (Reasoned Extension)`  
  - `FAIL: Hallucinated Citation`  
  - `FAIL: Contradiction`

- ✅ **Multi-Chat UI (Case-Based)**
  - Multiple independent chats (like separate legal cases)
  - Auto-named based on the first message
  - Switch between chats anytime

- ✅ **Interactive Knowledge Graph Inspector**
  - Check if specific sections or legal concepts exist in the graph.

---

## 🧠 System Architecture



User Query
↓
HyDE Generation (Hypothetical Law)
↓
Semantic Embedding
↓
Vector Similarity Search on BNS Knowledge Graph
↓
Top-K Legal Nodes Retrieved
↓
LLM Draft Answer
↓
LLM Legal Refinement (using graph as ground truth)
↓
Critic Verification
↓
Final Response + Legal Evidence


---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **LLM**: Google Gemini API  
- **Embeddings**: Google Text-Embedding Model  
- **Graph**: NetworkX + GraphML  
- **Vector Search**: Cosine similarity on in-memory embeddings  
- **Async Architecture**: Safe async + threading (Streamlit compatible)

---

## 🖥️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/novanotfound/bns_ai.git
cd bns_ai

2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Set up environment variables

Create a .env file:

GOOGLE_API_KEY=your_gemini_api_key_here

5. Run the application
streamlit run app.py

💬 Example Queries

“My neighbor cut a tree in my yard while I was out. Is it punishable?”

“What is the difference between theft and robbery under BNS?”

“Explain Section 303 with intuitive examples.”

“What happens if someone destroys government property?”

🧪 Why this project is different

Most legal AI tools:

Either hallucinate

Or are just search engines.

This system:
✅ Uses structured knowledge
✅ Verifies legal claims
✅ Separates reasoning from facts
✅ Clearly shows evidence

It treats law like a graph problem + reasoning problem, not just a text problem.

👨‍💻 Author

Priyanshu Janrao
B.Tech Computer Science Engineering
Project: BNS Legal AI Assistant
GitHub: https://github.com/novanotfound

📌 Project Status

Currently improving:

Case summarization per chat

Improved section linking

Performance optimization

Open to collaborations and suggestions ✨



