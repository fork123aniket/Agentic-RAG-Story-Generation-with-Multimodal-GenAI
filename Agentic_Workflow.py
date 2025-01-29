import os
import json
from PIL import Image
import pyttsx3
import io
import hashlib

import streamlit as st
import cloudinary
import cloudinary.uploader
import cloudinary.api

from langgraph import Stategraph
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain


# Step 1: Load and preprocess the VIST dataset
def preprocess_vist(json_file, image_dir):
    """
    Load and preprocess the VIST dataset to create (image, text) pairs.
    Args:
        json_file: Path to the VIST JSON file.
        image_dir: Path to the directory containing images.
    Returns:
        A list of tuples [(image_path, text), ...].
    """
    data = json.load(open(json_file))
    image_text_pairs = []

    for story in data["annotations"]:
        photo_ids = story["photo_ids"]  # IDs of images in the story
        sentences = story["story"]  # Corresponding sentences for each photo
        for photo_id, sentence in zip(photo_ids, sentences):
            image_path = os.path.join(image_dir, f"{photo_id}.jpg")
            if os.path.exists(image_path):
                image_text_pairs.append((image_path, sentence))

    return image_text_pairs

# Step 3: Store embeddings in LangChain's Chroma vectorstore
def store_embeddings_in_chroma(image_text_pairs, persist_directory):
    """
    Store embeddings in LangChain's Chroma vectorstore.
    Args:
        embeddings: List of (embedding, metadata) pairs.
        persist_directory: Path to store the Chroma vectorstore.
    """

    meta_data = []
    for image_path, text in image_text_pairs:
        metadata = {"image_path": image_path, "text": text}
        meta_data.append((metadata))

    documents = [
        Document(page_content=metadata["text"], metadata={"image_path": metadata["image_path"]})
        for metadata in meta_data
    ]

    hf_embeddings = HuggingFaceEmbeddings(model_name="openai/clip-vit-base-patch32")

    # Create Chroma vectorstore
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=hf_embeddings,
        persist_directory=persist_directory
    )

    vectorstore.persist()

# Function to hash an image
def hash_image(image):
    """Generate a hash for the image to detect changes."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

# Function to speak text
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Callback function for button press
def handle_speak(story):
    bot_message = story[0]["content"]
    st.session_state.speak_triggered = False
    speak_text(bot_message)

# Function to upload an image and return a URL or encoded string
def get_image_url(image_path):
    # Configure Cloudinary with your credentials
    cloudinary.config(
        cloud_name='cloud_name',  # Replace with your Cloudinary cloud name
        api_key='api_key',        # Replace with your Cloudinary API key
        api_secret='secret_key'   # Replace with your Cloudinary API secret
)

    # Upload the image to Cloudinary
    upload_result = cloudinary.uploader.upload(image_path)
    
    # Get the public URL of the uploaded image
    image_url = upload_result.get('secure_url')
    if image_url:
        print(f"Image uploaded successfully! Public URL: {image_url}")
        return image_url
    else:
        print("Failed to upload image.")
        return None

def agentic_workflow(image_url):
    # Initialize Stategraph
    workflow = Stategraph()

    # Step 1: Define nodes
    workflow.add_node("Start", lambda: print("Workflow started"))

    # LLM Node
    story_llm = ChatOpenAI(model="OpenGVLab/InternVL2-40B", temperature=0.7, max_tokens=512, timeout=None, max_retries=2)

    def llm_node(context):
        print("LLM Node executed")
        return context
    workflow.add_node("LLM", llm_node)

    # Tool Node
    chroma_vectorstore = Chroma(persist_directory="path_to_chroma_vectorstore")
    hf_embeddings = HuggingFaceEmbeddings(model_name="openai/clip-vit-base-patch32")

    def tool_node(context):
        print("Tool Node executed")
        # Generate embedding for the user-uploaded image
        image_embedding = hf_embeddings.embed_documents([context["image_path"]])

        # Search for relevant documents in the vectorstore
        relevant_docs = chroma_vectorstore.similarity_search_by_vector(image_embedding[0], k=2)
        if relevant_docs:
            context["retrieved_docs"] = relevant_docs
            return "Retriever"
        else:
            return "Generator"
    workflow.add_node("Tools", tool_node)

    # Retriever Node
    def retriever_node(context):
        print("Retriever Node executed")
        # Store retrieved embeddings for story generation
        context["retrieved_embeddings"] = [doc.page_content for doc in context["retrieved_docs"]]
        return context
    workflow.add_node("Retriever", retriever_node)

    # Generator Node
    def generator_node(context):
        print("Generator Node executed")
        
        """Generate a narrative tying together the image and related story texts (if any)."""

        # Extract textual information from retrieved embeddings
        if "retrieved_embeddings" in context:
            retrieved_texts = ";".join([embedding["text"] for embedding in context["retrieved_embeddings"]])

            prompt_template = """
            You are a creative storyteller. Given the following image and related story texts (might be relevant and helpful for you to generate the desired output), create an engaging and coherent narrative:

            {image}
            {external_relevant_context}

            Use external relevant context (separated by a semicolon ";") if necessary to enhance the story, ensuring it remains engaging and meaningful.
            """
            
            prompt = ChatPromptTemplate.from_template(prompt_template).format(
                image=image_url,
                external_relevant_context=retrieved_texts
            )
        else:
            prompt_template = """
            Given the following image, use your imagination and reasoning abilities to craft an engaging and immersive story. 
            Think about the environment, the potential characters, the emotions that might arise from this scene, and any 
            unexpected events that could unfold. Be creative, and bring the story to life by describing the atmosphere, 
            characters, and events in a detailed and captivating way.

            {image}
            """

            prompt = ChatPromptTemplate.from_template(prompt_template).format(
                image=image_url)
            
        chain = LLMChain(llm=story_llm, prompt=prompt)
        story = chain.run()
        context["story"] = story
        return context
    workflow.add_node("Generator", generator_node)

    # End Node
    def end_node(context):
        st.success("Story generated successfully!")
        st.session_state.speak_triggered = True
        st.text_area("Generated Story:", context["story"])
        st.session_state.chat_history.append(
            {"role": "assistant", "content": context["story"]}
        )
        st.button(
            "Speak",
            on_click=handle_speak,
            args=(st.session_state.chat_history,)
        )
        return context
    workflow.add_node("End", end_node)

    # Step 2: Add edges
    workflow.add_edge("Start", "LLM")
    workflow.add_edge("LLM", "Tools")
    workflow.add_conditional_edge("Tools", "Retriever", condition=lambda context: "retrieved_docs" in context and context["retrieved_docs"])
    workflow.add_conditional_edge("Tools", "Generator", condition=lambda context: "retrieved_docs" not in context)
    workflow.add_edge("Retriever", "Generator")
    workflow.add_edge("Generator", "End")
    
    return workflow

# Streamlit App
def run_streamlit_app():
    
    st.title("Visual Storytelling Generator")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "image_hash" not in st.session_state:
        st.session_state.image_hash = None
    if "speak_triggered" not in st.session_state:
        st.session_state.speak_triggered = False  # To track which message's Speak button was pressed

    # Upload image files
    uploaded_image = st.file_uploader("Upload images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Save the image to a file
        image.save("user_uploaded_photo.png")

        # Get image URL or encode image
        image_url = get_image_url("user_uploaded_photo.png")

        # Check if the image has changed
        current_image_hash = hash_image(image)
        if st.session_state.image_hash is None:
            st.session_state.image_hash = current_image_hash
        elif st.session_state.image_hash != current_image_hash:
            st.session_state.chat_history = []  # Reset chat history
            st.session_state.image_hash = current_image_hash
            st.warning("Image has changed!")

        # Run Agentic Workflow
        with st.spinner("Running the Agentic AI Workflow to Generate the Story..."):
            context = {
                "image_path": image_url
            }
            workflow = agentic_workflow(image_url)
            workflow.run(context)

# Display Story
if not st.session_state.speak_triggered:
    st.text_area("Generated Story:", st.session_state.chat_history[0]["content"])
    st.button(
        "Speak",
        on_click=handle_speak,
        args=(st.session_state.chat_history,)
    )

if __name__ == "__main__":

    # Paths
    vist_json = "path_to_vist_annotations.json"
    image_dir = "path_to_vist_images"
    persist_dir = "path_to_chroma_vectorstore"

    with st.spinner("Loading the Dataset, Generating and Saving the Embeddings..."):
        # Preprocess the dataset
        image_text_pairs = preprocess_vist(vist_json, image_dir)

        # Store embeddings in Chroma vectorstore
        store_embeddings_in_chroma(image_text_pairs, persist_dir)

        st.success("Embeddings stored in Chroma vectorstore successfully!")

    run_streamlit_app()
