import streamlit as st
import nltk
import base64
from app import extract_text_from_pdf, process_text, calculate_similarity
import streamlit as st
import base64


# Ensure required NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

st.sidebar.markdown(
    '''
This measure is a judgment of orientation and not magnitude: it measures whether two vectors are pointing in roughly the same direction.

Cosine Similarity Formula:
''')

st.sidebar.latex(r'''
(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\| \vec{A} \| \| \vec{B} \|}
    ''')

st.sidebar.markdown(
'''
The result will range from -1 to 1 where:
- 1 indicates the vectors are identical.
- 0 indicates orthogonality (no similarity).
- -1 indicates the vectors are diametrically opposed.
''')

def displayPDF(uploaded_file):
    
    # Read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')

    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700" type="application/pdf"></iframe>'

    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit app layout
st.title('Resume Scoring App')

resume_file = st.file_uploader("Upload your resume (PDF format)", type=['pdf'])
job_desc_file = st.file_uploader("Upload job description (PDF format)")

if resume_file:
    st.write("Resume Uploaded Successfully!")
    displayPDF(resume_file)


if st.button('Score Resume'):
    if resume_file and job_desc_file:
        resume_text = extract_text_from_pdf(resume_file)
        job_desc_text = extract_text_from_pdf(job_desc_file)

        processed_resume_text = process_text(resume_text)
        processed_job_desc_text = process_text(job_desc_text)

        score = calculate_similarity(processed_resume_text, processed_job_desc_text)
        st.write(f"Similarity Score: {score:.2f}")
    else:
        st.write("Please upload both resume and job description PDF files.")
