import streamlit as st
from groq import Groq
import base64
import os
from PIL import Image
import io
import json

# Page configuration
st.set_page_config(
    page_title="Groq Vision & OCR",
    page_icon="👁️",
    layout="wide"
)

# Set API key
GROQ_API_KEY = "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"

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Models
VISION_MODELS = {
    "Llama 4 Scout (Fast & Efficient)": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Llama 4 Maverick (Advanced)": "meta-llama/llama-4-maverick-17b-128e-instruct"
}

# Convert image to base64
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# Image analysis function
def analyze_image(image_data, prompt, model, use_json=False, is_url=False):
    try:
        # Message content
        content = [{"type": "text", "text": prompt}]
        
        if is_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_data}
            })
        else:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            })
        
        # API request
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 1,
            "max_completion_tokens": 2048,
            "top_p": 1,
            "stream": False,
        }
        
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}
        
        completion = client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"Error: {str(e)}"

# Title
st.title("👁️ Groq Vision & OCR")
st.markdown("**Image analysis, OCR, text extraction, and more!**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    selected_model_name = st.selectbox(
        "Choose Model",
        list(VISION_MODELS.keys()),
        help="Llama 4 Scout is fast, Maverick is more advanced"
    )
    model = VISION_MODELS[selected_model_name]
    
    st.divider()
    
    # Analysis mode
    st.subheader("📋 Analysis Mode")
    analysis_mode = st.radio(
        "Mode",
        ["Custom Question", "OCR (Extract Text)", "General Analysis", "JSON Output"],
        help="Choose how to analyze the image"
    )
    
    st.divider()
    
    # Limits info
    with st.expander("ℹ️ Limits"):
        st.markdown("""
        **📏 Image Limits:**
        - URL: Max 20MB  
        - Base64: Max 4MB  
        - Resolution: 33 megapixels  
        - Max images per request: 5  
        
        **🧠 Model Features:**
        - 128K context  
        - Multi-language support  
        - JSON mode  
        - Tool use capability  
        """)

# Tabs
tab1, tab2 = st.tabs(["📤 Upload Image", "🌐 Use URL"])

with tab1:
    st.subheader("📷 Upload Images")
    
    uploaded_files = st.file_uploader(
        "Upload images (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="You can upload up to 5 images"
    )
    
    if uploaded_files:
        if len(uploaded_files) > 5:
            st.error("❌ You can upload a maximum of 5 images!")
        else:
            # Display images
            cols = st.columns(min(len(uploaded_files), 3))
            for idx, uploaded_file in enumerate(uploaded_files):
                with cols[idx % 3]:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
                    
                    file_size = uploaded_file.size / (1024 * 1024)  
                    st.caption(f"Size: {file_size:.2f} MB")
            
            st.divider()
            
            # Prompt area
            if analysis_mode == "Custom Question":
                user_prompt = st.text_area(
                    "Enter your question",
                    value="What is in this image?",
                    height=100
                )
            elif analysis_mode == "OCR (Extract Text)":
                user_prompt = "Extract and read all text from this image. Write the text exactly as it appears."
                st.info("💡 OCR mode: All text in the image will be extracted.")
            elif analysis_mode == "General Analysis":
                user_prompt = "Analyze this image in detail. What do you see? Describe important elements."
                st.info("💡 General analysis mode activated.")
            else:
                user_prompt = "List all information in this image in JSON format, including objects, colors, text, and positions."
                st.info("💡 JSON mode: Structured output will be generated.")
            
            # Analyze button
            if st.button("🔍 Start Analysis", type="primary", use_container_width=True):
                with st.spinner("🧠 Analyzing image..."):
                    for idx, uploaded_file in enumerate(uploaded_files):
                        st.subheader(f"📊 Result: {uploaded_file.name}")
                        
                        uploaded_file.seek(0)
                        base64_image = encode_image(uploaded_file)
                        
                        result = analyze_image(
                            base64_image,
                            user_prompt,
                            model,
                            use_json=(analysis_mode == "JSON Output"),
                            is_url=False
                        )
                        
                        if analysis_mode == "JSON Output":
                            try:
                                json_result = json.loads(result)
                                st.json(json_result)
                            except:
                                st.code(result, language="json")
                        else:
                            st.markdown(result)
                        
                        if idx < len(uploaded_files) - 1:
                            st.divider()

with tab2:
    st.subheader("🌐 Image Analysis via URL")
    
    image_url = st.text_input(
        "Enter image URL",
        placeholder="https://example.com/image.jpg",
        help="Max 20MB image"
    )
    
    if image_url:
        try:
            st.image(image_url, caption="Preview", use_container_width=True)
        except:
            st.warning("⚠️ Could not display preview, but analysis can still be performed.")
        
        st.divider()
        
        if analysis_mode == "Custom Question":
            user_prompt = st.text_area(
                "Enter your question",
                value="What is in this image?",
                height=100,
                key="url_prompt"
            )
        elif analysis_mode == "OCR (Extract Text)":
            user_prompt = "Extract and read all text from this image. Write the text exactly as it appears."
            st.info("💡 OCR mode activated.")
        elif analysis_mode == "General Analysis":
            user_prompt = "Analyze this image in detail. What do you see? Describe important elements."
            st.info("💡 General analysis mode activated.")
        else:
            user_prompt = "List all information in this image in JSON format, including objects, colors, text, and positions."
            st.info("💡 JSON mode activated.")
        
        if st.button("🔍 Analyze URL", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing image..."):
                result = analyze_image(
                    image_url,
                    user_prompt,
                    model,
                    use_json=(analysis_mode == "JSON Output"),
                    is_url=True
                )
                
                st.subheader("📊 Analysis Result")
                
                if analysis_mode == "JSON Output":
                    try:
                        json_result = json.loads(result)
                        st.json(json_result)
                    except:
                        st.code(result, language="json")
                else:
                    st.markdown(result)

# Footer
st.divider()

# Examples
with st.expander("💡 Example Use Cases"):
    st.markdown("""
    ### 📋 OCR (Text Extraction)
    - Reading documents, invoices, receipts  
    - Handwriting recognition  
    - Extracting tables and form data  
    
    ### 🖼️ Image Analysis
    - Product recognition  
    - Scene and object detection  
    - Color and composition analysis  
    
    ### 🌍 Multi-language Support
    - Reading text in different languages  
    - Processing international documents  
    
    ### 📊 JSON Output
    - Structured data extraction  
    - Catalog automation  
    - Database integration  
    """)

st.markdown("""
### 📚 About
This application uses **Groq API** and **Llama 4 Vision** models:
- ⚡ **Llama 4 Scout**: Fast and efficient image understanding  
- 🚀 **Llama 4 Maverick**: More accurate and advanced  
- 🌐 **128K Context Window**  
- 🔧 **JSON Mode**  
- 💬 **Multi-turn image conversations**  

**Powered by Groq – Ultra-fast inference!**
""")
