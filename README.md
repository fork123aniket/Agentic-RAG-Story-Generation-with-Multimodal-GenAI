# 📖 **Agentic RAG Storytelling with Multimodal AI**  

### **Transform Images into Engaging Narratives with Generative AI**  

## 🚀 Overview  
This project implements an **Agentic RAG (Retrieval-Augmented Generation) workflow** using **LangChain, LangGraph, and multimodal AI** to generate meaningful and engaging stories from user-uploaded images. The system seamlessly integrates **retrieval-based reasoning** with **generative AI capabilities**, leveraging **LLMs, vector search, and vision models** to produce immersive narratives.  

---

## 🎯 **Why Agentic RAG?**  
This project demonstrates how **LLMs, retrieval systems, and vision models** can collaborate to generate dynamic and **engaging AI-generated stories**. The **agentic workflow** makes it possible to balance **creativity and factuality**, ensuring a seamless storytelling experience.

---

## ✨ Key Features  
🔹 **Multimodal Input Handling**: Accepts **image uploads** and processes them for storytelling.  
🔹 **Retrieval-Augmented Generation (RAG)**: Enhances story coherence using **retrieved knowledge**.  
🔹 **Agentic AI Workflow**: Dynamically selects **retriever** or **generator** based on available data.  
🔹 **LLM-Driven Storytelling**: Generates **rich, context-aware** narratives using **GPT-4o**.  
🔹 **Chroma Vector Search**: Stores and retrieves **image-text embeddings** for enhanced relevance.  
🔹 **Cloud Integration**: Supports **Cloudinary** for image hosting and efficient handling.  
🔹 **Streamlit UI**: Interactive interface for seamless user interaction.  

---

## 🛠 **Technology Stack**  

This project leverages cutting-edge **LLMs, multimodal AI, and retrieval-based techniques** to ensure seamless and engaging storytelling.  

| **Category**       | **Technology Used** |
|--------------------|--------------------|
| 🤖 **LLM**         | InternVL2 - 40B |
| 🖼 **Vision Model** | CLIP (Contrastive Language-Image Pretraining) |
| 🔍 **Retriever**    | ChromaDB Vector Search |
| 📜 **RAG Framework** | LangChain, LangGraph |
| 🌐 **Web Interface** | Streamlit |
| ☁️ **Cloud Storage** | Cloudinary API |
| 🎤 **Text-to-Speech** | pyttsx3 |

These technologies work together to **process images, retrieve relevant context, and generate rich, meaningful narratives**.  

---

## 🎯 **Agentic RAG Gen AI Workflow**  

This project follows an **Agentic RAG (Retrieval-Augmented Generation) framework**, meaning the system **autonomously** decides when to retrieve external knowledge and when to generate stories based on the input.  

### **How It Works**  
1️⃣ **Image Upload**: Users provide an image via the **Streamlit UI**.  
2️⃣ **Embedding Generation**: The image is processed using **Hugging Face CLIP** embeddings.  
3️⃣ **Retrieval Decision**:  
   - If relevant **stories exist**, the retriever fetches them for context.  
   - If no prior data is found, the system **generates a story from scratch**.  
4️⃣ **LLM-Based Generation**: Uses **GPT-4o** to craft a compelling story.  
5️⃣ **User Display**: The generated story is displayed in the UI.  
6️⃣ **Audio Narration** – Converts generated stories into speech.  

This approach ensures **optimal use of existing knowledge** while allowing for **creative storytelling** when necessary.  

---

## 🛠 **Workflow Architecture**  

The diagram below illustrates the **modular AI workflow** used in this project.  

![Visual Storytelling Workflow](image.png)  

### **Workflow Breakdown**  
- **Start** → Initializes the workflow.  
- **LLM Node** → Processes input data and decides next steps.  
- **Tools Node** → Determines whether to retrieve past stories or generate new ones.  
- **Retriever Node** → If prior knowledge exists, retrieves related image-text pairs.  
- **Generator Node** → Uses the **LLM** to create an engaging story.  
- **End** → Displays the final generated narrative to the user.  

This **state-driven agentic AI approach** ensures that each decision maximizes the **relevance and creativity** of the generated content.  

---

## 📌 **Installation & Usage**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/fork123aniket/VLM-powered-Multimodal-Conversational-AI-Bot.git
cd VLM-powered-Multimodal-Conversational-AI-Bot
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Set Up Cloudinary for Image Hosting**  
Replace the placeholders in `Agentic_Workflow.py`:  
```python
cloudinary.config(
    cloud_name='your_cloud_name',
    api_key='your_api_key',
    api_secret='your_secret_key'
)
```

### **4️⃣ Run the Streamlit App**  
```bash
streamlit run Agentic_Workflow.py
```

### **5️⃣ Upload an Image & Enjoy AI-Powered Storytelling!**  

---

## 📂 **Project Structure**  

The project is structured as follows:  

```
📦 Agentic_RAG_Storytelling
├── 📜 Agentic_Workflow.py      # Core agentic workflow implementation
├── 📜 requirements.txt         # Dependencies list
├── 📜 README.md                # Project documentation