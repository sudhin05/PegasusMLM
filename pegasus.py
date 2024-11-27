import streamlit as st
import os
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from io import BytesIO

# URL for PEGASUS model information
url_policy = 'https://example.com/hr-policy-guidelines'
st.markdown("# HR Policy Document Summarizer")
st.markdown(
    '''Human Resource (HR) policies are critical for defining the guidelines and procedures that govern employee behavior and organizational practices.
    This summarizer uses advanced NLP techniques to provide concise summaries of HR policy documents, helping HR professionals and employees quickly
    understand key points and updates. With the use of state-of-the-art models like PEGASUS, this tool offers a quick way to digest comprehensive
    policy documents into actionable insights. For more detailed information on the PEGASUS model, you can refer to [this resource](%s).''' % url_policy
)

st.header('''Upload your HR policy document below to receive a summarized version. The document should be a PDF with a size less than 200MB.
The speed of summarization may vary depending on the length and complexity of the document. Please upload one PDF at a time.''')

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload an HR Policy Document")

if uploaded_file:
    # Save the uploaded file temporarily
    with open(os.path.join("D:/Users/mahes/DTU_Societies/SIH", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully!")

st.markdown("# Policy Summary")

# Load PEGASUS model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")

if uploaded_file:
    with open(os.path.join("D:/Users/mahes/DTU_Societies/SIH", uploaded_file.name), 'rb') as fhandle:
        pdfReader = PyPDF2.PdfReader(fhandle)

        # Get number of pages using the updated method
        num_pages = len(pdfReader.pages)
        values = st.slider(
            'Select page range for summarization:',
            min_value=1, max_value=num_pages+1,
            value=(1, min(num_pages, 5)),  # Default to first 5 pages or less if document is shorter
            step=1
        )

        with st.spinner('Generating summary. Please wait...'):
            for x in range(values[0] - 1, values[1]):
                pagehandle = pdfReader.pages[x]  # Updated for new PyPDF2 API
                text = pagehandle.extract_text()

                if not text.strip():
                    st.warning(f"Page {x + 1} contains no extractable text.")
                    continue

                tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
                summary_ids = model.generate(tokens["input_ids"], max_length=150, min_length=40, length_penalty=2.0)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                st.markdown(f"### Page {x + 1}")
                st.text_area(label="", value=summary, placeholder="Summary will appear here", height=100, key=x + 1)
                st.markdown("")
