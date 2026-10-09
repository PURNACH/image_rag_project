import streamlit as st

from src.image_processing import preprocess_image
from src.ocr import extract_text
from src.text_processing import clean_text, chunk_text
from src.vector_store import upsert_chunks
from src.rag_pipeline import retrieve_context
from src.llm import generate_answer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Image RAG Question Answering",
    page_icon="🖼️",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🖼️ Image-Based RAG Question Answering System")

st.write(
    "Upload an image containing text, extract the information, "
    "store it in a vector database, and ask questions about it."
)


# --------------------------------------------------
# API KEY
# --------------------------------------------------

groq_api_key = st.text_input(
    "Enter your Groq API Key",
    type="password"
)

if groq_api_key:
    import os

    os.environ["GROQ_API_KEY"] = groq_api_key


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)


# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    st.subheader("Uploaded Image")

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Process Image"):

        try:

            # ------------------------------------------
            # Read image
            # ------------------------------------------

            image_bytes = uploaded_file.getvalue()

            # ------------------------------------------
            # Image preprocessing
            # ------------------------------------------

            with st.spinner("Preprocessing image..."):

                processed_image = preprocess_image(
                    image_bytes
                )

            st.success("Image preprocessing completed.")

            # ------------------------------------------
            # OCR
            # ------------------------------------------

            with st.spinner("Extracting text using OCR..."):

                extracted_text = extract_text(
                    processed_image
                )

            if not extracted_text:

                st.warning(
                    "No text was detected in the image."
                )

                st.stop()

            # ------------------------------------------
            # Clean text
            # ------------------------------------------

            cleaned_text = clean_text(
                extracted_text
            )

            # ------------------------------------------
            # Display OCR text
            # ------------------------------------------

            st.subheader("📄 Extracted Text")

            st.text_area(
                "OCR Output",
                cleaned_text,
                height=250
            )

            # ------------------------------------------
            # Chunk text
            # ------------------------------------------

            chunks = chunk_text(
                cleaned_text,
                chunk_size=100,
                overlap=20
            )

            st.write(
                f"Created **{len(chunks)}** text chunks."
            )

            # ------------------------------------------
            # Store in ChromaDB
            # ------------------------------------------

            with st.spinner(
                "Creating embeddings and storing in ChromaDB..."
            ):

                upsert_chunks(chunks)

            st.success(
                "Text successfully stored in ChromaDB!"
            )

            # Save session state
            st.session_state["processed"] = True

        except Exception as e:

            st.error(
                f"Error while processing image: {str(e)}"
            )


# --------------------------------------------------
# QUESTION ANSWERING
# --------------------------------------------------

if st.session_state.get("processed", False):

    st.divider()

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Enter your question about the uploaded image:"
    )

    if st.button("🤖 Ask Question"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        elif not groq_api_key:

            st.warning(
                "Please enter your Groq API key first."
            )

        else:

            try:

                # --------------------------------------
                # Retrieve relevant context
                # --------------------------------------

                with st.spinner(
                    "Searching relevant information..."
                ):

                    context = retrieve_context(
                        question,
                        top_k=3
                    )

                if not context:

                    st.warning(
                        "No relevant information found."
                    )

                else:

                    # ----------------------------------
                    # Generate answer
                    # ----------------------------------

                    with st.spinner(
                        "Generating answer..."
                    ):

                        answer = generate_answer(
                            question,
                            context
                        )

                    st.subheader("🤖 Answer")

                    st.write(answer)

                    # ----------------------------------
                    # Show retrieved context
                    # ----------------------------------

                    with st.expander(
                        "View Retrieved Context"
                    ):

                        st.write(context)

            except Exception as e:

                st.error(
                    f"Error while answering: {str(e)}"
                )