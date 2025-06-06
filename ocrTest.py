from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1
import os
from PyPDF2 import PdfReader, PdfWriter
import tempfile
import time
from keybert import KeyBERT

# TODO(developer): Create a processor of type "OCR_PROCESSOR".

# TODO(developer): Update and uncomment these variables before running the sample.
project_id = "cs-poc-ofgvbxztycxqy0e3prnjr9t"

# Processor ID as hexadecimal characters.
# Not to be confused with the Processor Display Name.
processor_id = "cc9c0fd7e3ae815b"

# Processor location. For example: "us" or "eu".
location = "us"

# Path for file to process.
# file_path = "G:\GPTs\ThothStream\TD89-1.pdf"

# Set `api_endpoint` if you use a location other than "us".
opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

# Initialize Document AI client.
client = documentai_v1.DocumentProcessorServiceClient(client_options=opts)

# Get the Fully-qualified Processor path.
full_processor_name = client.processor_path(project_id, location, processor_id)

# Get a Processor reference.
request = documentai_v1.GetProcessorRequest(name=full_processor_name)
processor = client.get_processor(request=request)

# `processor.name` is the full resource name of the processor.
# For example: `projects/{project_id}/locations/{location}/processors/{processor_id}`
print(f"Processor Name: {processor.name}")

# Helper function to split PDF into chunks of max_pages
def split_pdf(input_pdf_path, max_pages=15):
    reader = PdfReader(input_pdf_path)
    total_pages = len(reader.pages)
    chunks = []
    for start in range(0, total_pages, max_pages):
        writer = PdfWriter()
        for i in range(start, min(start + max_pages, total_pages)):
            writer.add_page(reader.pages[i])
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        with open(temp_file.name, 'wb') as f:
            writer.write(f)
        chunks.append(temp_file.name)
    return chunks

# --- Generate YAML frontmatter ---
def extract_title(text):
    # Try to find a line that looks like a title (e.g., starts with # or is all caps and not too long)
    lines = text.splitlines()
    for line in lines:
        if line.strip().startswith('# '):
            return line.strip('# ').strip()
    # Fallback: first non-empty, reasonably short, mostly capitalized line
    for line in lines:
        if line.strip() and len(line.strip()) < 80 and line.strip().isupper():
            return line.strip()
    # Fallback: first non-empty line
    for line in lines:
        if line.strip():
            return line.strip()
    return "Untitled"

# --- Batch processing for multiple PDFs in a directory ---

def process_pdf(file_path, output_dir, processor, client):
    input_basename = os.path.splitext(os.path.basename(file_path))[0]
    output_md_path = os.path.join(output_dir, f"{input_basename}.md")
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    if total_pages > 15:
        print(f"PDF has {total_pages} pages, splitting into chunks of 15...")
        pdf_chunks = split_pdf(file_path, max_pages=15)
    else:
        pdf_chunks = [file_path]
    all_text = ""
    for idx, chunk_path in enumerate(pdf_chunks):
        print(f"Processing chunk {idx+1}/{len(pdf_chunks)}: {chunk_path}")
        with open(chunk_path, "rb") as image:
            image_content = image.read()
        raw_document = documentai_v1.RawDocument(
            content=image_content,
            mime_type="application/pdf",
        )
        request = documentai_v1.ProcessRequest(name=processor.name, raw_document=raw_document)
        result = client.process_document(request=request)
        document = result.document
        all_text += document.text + "\n"
        time.sleep(1)  # polite pause between requests
    # --- Extract and display keywords from combined OCR text ---
    ocr_text = all_text
    kw_model = KeyBERT()
    # Use only the keyword (not the score) for tags in frontmatter
    keywords = kw_model.extract_keywords(ocr_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
    tags = [repr(kw) for kw, _ in keywords[:20]]  # Top 5 keywords as tags, quoted for YAML
    print("\nTop keywords found in the combined document:")
    for kw, score in keywords:
        print(f"- {kw} (score: {score:.2f})")
    # --- Generate YAML frontmatter ---
    title = extract_title(ocr_text)
    source = os.path.basename(file_path)
    frontmatter = f"""---\ntitle: {title}\ntags: [{', '.join(tags)}]\nsource: {source}\n---\n\n"""
    # Save combined OCR text and keywords to Markdown with YAML frontmatter
    with open(output_md_path, "w", encoding="utf-8") as md_file:
        md_file.write(frontmatter)
        md_file.write("# Combined OCR Output\n\n")
        md_file.write("## Top Keywords\n")
        for kw, score in keywords:
            md_file.write(f"- {kw} (score: {score:.2f})\n")
        md_file.write("\n")
        md_file.write(ocr_text)
    print(f"Combined OCR results and keywords saved to {output_md_path}")

# --- Main batch logic ---
input_dir = r"G:\Thoth\TempleDoors"  # Directory with PDFs to process
output_dir = r"G:\Thoth\ThothStream\v2.1\templeDoorsMD"
os.makedirs(output_dir, exist_ok=True)

pdf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
print(f"Found {len(pdf_files)} PDF(s) to process.")
for pdf_file in pdf_files:
    print(f"\n---\nProcessing: {pdf_file}")
    process_pdf(pdf_file, output_dir, processor, client)