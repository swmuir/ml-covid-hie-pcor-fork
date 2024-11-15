'''import base64
import torch
import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline, AutoModel
import numpy as np
import pandas as pd
import re
import spacy
import sklearn
from sklearn.random_projection import GaussianRandomProjection

print("torch version:", torch.__version__)
print("transformers version:", transformers.__version__)
print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
print("spacy version:", spacy.__version__)
print("sklearn version:", sklearn.__version__)												 
						 
																												   
#install spacy and med7 model before this program
#pip install spacy==2.3.5
#pip install https://huggingface.co/kormilitzin/en_core_med7_trf/resolve/main/en_core_med7_trf-any-py3-none-any.whl

# Load spaCy model for med7
nlp = spacy.load("en_core_med7_trf")

# Load BioBERT model and tokenizer
biobert_model_name = "emilyalsentzer/Bio_ClinicalBERT"
device = "cuda" if torch.cuda.is_available() else "cpu"
biobert_model = AutoModel.from_pretrained(biobert_model_name).to(device)
biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)

# Load RoBERTa model for deidentification
deid_model_name = "obi/deid_roberta_i2b2"
deid_model = AutoModelForTokenClassification.from_pretrained(deid_model_name)
deid_tokenizer = AutoTokenizer.from_pretrained(deid_model_name).to(device)
deid_pipeline = pipeline("ner", model=deid_model, tokenizer=deid_tokenizer, aggregation_strategy="simple")

def deidentify_text(text):

    print(text)                            			   
    ner_results = deid_pipeline(text)
    deidentified_text = text
												 
    for entity in ner_results:
        if entity['entity_group'] in ['PATIENT', 'DOCTOR', 'HOSPITAL', 'DATE', 'ID', 'AGE', 'LOC', 'PATORG', 'EMAIL', 'OTHERPHI']:
            entity_text = entity['word']
            deidentified_text = deidentified_text.replace(entity_text, '[REDACTED]')
    
															
    patterns = {
        r'\b\d{3}-\d{2}-\d{4}\b': '[REDACTED]',  # SSNs
        r'\b\d{2}/\d{2}/\d{4}\b': '[REDACTED]',  # Dates in MM/DD/YYYY format
        r'\b\d{5}\b': '[REDACTED]',  # Zip codes
        r'\b\d{3}-\d{3}-\d{4}\b': '[REDACTED]',  # Phone numbers
        r'\b\d{10}\b': '[REDACTED]',  # Phone numbers without dashes
        r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b': '[REDACTED]',  # Emails
        r'\b\d{1,4}-\d{1,4}-\d{1,4}\b': '[REDACTED]',  # Medical record numbers
        r'\b(?:[A-Fa-f0-9]{2}:){5}[A-Fa-f0-9]{2}\b': '[REDACTED]',  # MAC addresses
        r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b': '[REDACTED]',  # IPv4 addresses
        r'\b(?:[0-9a-fA-F]{4}::?){2,7}\b': '[REDACTED]',  # IPv6 addresses
        r'\b(?:\d{4}[- ]){3}\d{4}\b': '[REDACTED]',  # Credit card numbers
        r'\b[1-9]\d{2}-\d{2}-\d{4}\b': '[REDACTED]',  # Alternate SSNs
        r'\b[A-Z]{3}-\d{3}-\d{3}\b': '[REDACTED]',  # License plate numbers
        r'\b\d{9}\b': '[REDACTED]',  # Generic ID numbers
        r'\b\d{4}\b': '[REDACTED]',  # Year
									  
    }
    
    for pattern, replacement in patterns.items():
        deidentified_text = re.sub(pattern, replacement, deidentified_text, flags=re.IGNORECASE)
		
    
    return deidentified_text

def chunk_text(text, chunk_size=512):
    chunks = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))
        chunk = text[start_idx:end_idx]
        if end_idx < len(text) and not re.match(r"\b\w+\b$", chunk):
            chunk += " " + text[end_idx]
            end_idx += 1
        chunks.append(chunk)
        start_idx = end_idx
    return chunks

def generate_embeddings(text, model, tokenizer):
    chunks = chunk_text(text)
    embeddings = []
    for chunk in chunks:
        encoded_input = tokenizer(chunk, return_tensors="pt", padding="max_length", truncation=True)
						   
        encoded_input = encoded_input.to(device)
        with torch.no_grad():
            output = model(**encoded_input)
            last_hidden_state = output.last_hidden_state
            chunk_embedding = torch.mean(last_hidden_state, dim=1)
            embeddings.append(chunk_embedding.cpu().detach().numpy())
    if embeddings:
        return np.mean(np.concatenate(embeddings), axis=0)
    else:
        return np.array([])

def extract_medical_entities(text):
    doc = nlp(text)
    medical_entities = " ".join([ent.text for ent in doc.ents])
    return medical_entities if medical_entities else "[EMPTY]"

# Load the CSV file
file_path = '/oncmloutput/cqlplanfile.csv'
df = pd.read_csv(file_path, encoding='cp1252',skip_blank_lines=False, keep_default_na=False, na_filter=False)

# Initialize lists to store embeddings
note_embeddings = []
med_embeddings = []


# Process each note
for note in df['Notes'].astype(str).replace(r'^\s*$', "[EMPTY]", regex=True).fillna('[EMPTY]'):#added [empty] to handle empty notes:#.replace(r'^\s*$', "[EMPTY]", regex=True).fillna('[EMPTY]'):#added [empty] to handle empty notes
    # Deidentify the note
    print(note)
    deidentified_note = deidentify_text(note)
    print(deidentified_note)
    # Generate embeddings for the deidentified note
    note_embedding = generate_embeddings(deidentified_note, biobert_model, biobert_tokenizer)
    
    note_embeddings.append(note_embedding.tolist() if note_embedding.size > 0 else [])
    
    # Extract medical entities using med7
    medical_entities = extract_medical_entities(deidentified_note)
    
    
    # Generate embeddings for the extracted medical entities
    med_embedding = generate_embeddings(medical_entities, biobert_model, biobert_tokenizer)
    
    med_embeddings.append(med_embedding.tolist() if med_embedding.size > 0 else [])

# Add embeddings to the DataFrame as single columns
#df['Note_Embeddings'] = note_embeddings
#df['Med_Embeddings'] = med_embeddings

# Convert lists back to numpy arrays for projection
note_embeddings_np = np.array(note_embeddings)
med_embeddings_np = np.array(med_embeddings)

# Apply GaussianRandomProjection to both sets of embeddings
transformer = GaussianRandomProjection(n_components=768, random_state=42)

projected_note_embeddings = transformer.fit_transform(note_embeddings_np)
projected_med_embeddings = transformer.fit_transform(med_embeddings_np)

# Add the projected embeddings back to the DataFrame
df['Projected_Note_Embeddings'] = [list(embedding) for embedding in projected_note_embeddings]
df['Projected_Med_Embeddings'] = [list(embedding) for embedding in projected_med_embeddings]
df = df.drop(columns=['Notes'])
# Save the updated DataFrame to a new CSV file
#output_file_path = '/oncmloutput/cqlplanfile_with_projected_embeddings.csv'
#df.to_csv(output_file_path, index=False)
output_file_path = '/oncmloutput/cqlplanfile_with_projected_embeddings.pkl'
df.reset_index(drop=True, inplace=True)
df.to_pickle(output_file_path)
output_file_path2 = '/oncmloutput/cqlplanfile_with_projected_embeddings.csv'
df.to_csv(output_file_path2, index=False)
 
print(f"Processed file saved to {output_file_path}")


print(f"Processed file saved to {output_file_path}")
'''
####################################################
import base64
import torch
import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline, AutoModel
import numpy as np
import pandas as pd
import re
import spacy
import sklearn
from sklearn.random_projection import GaussianRandomProjection
from concurrent.futures import ThreadPoolExecutor

print("torch version:", torch.__version__)
print("transformers version:", transformers.__version__)
print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
print("spacy version:", spacy.__version__)
print("sklearn version:", sklearn.__version__)												 
						 
# Load spaCy model for med7
nlp = spacy.load("en_core_med7_trf")

# Load BioBERT model and tokenizer
biobert_model_name = "emilyalsentzer/Bio_ClinicalBERT"
device = "cuda" if torch.cuda.is_available() else "cpu"
biobert_model = AutoModel.from_pretrained(biobert_model_name).to(device)
biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)

# Load RoBERTa model for deidentification
deid_model_name = "obi/deid_roberta_i2b2"
deid_model = AutoModelForTokenClassification.from_pretrained(deid_model_name).to(device)
deid_tokenizer = AutoTokenizer.from_pretrained(deid_model_name)
deid_pipeline = pipeline("ner", model=deid_model, tokenizer=deid_tokenizer, aggregation_strategy="simple", device=0)

def deidentify_text(text):
    ner_results = deid_pipeline(text)
    deidentified_text = text
    for entity in ner_results:
        if entity['entity_group'] in ['PATIENT', 'DOCTOR', 'HOSPITAL', 'DATE', 'ID', 'AGE', 'LOC', 'PATORG', 'EMAIL', 'OTHERPHI']:
            entity_text = entity['word']
            deidentified_text = deidentified_text.replace(entity_text, '[REDACTED]')
    patterns = {
        r'\b\d{3}-\d{2}-\d{4}\b': '[REDACTED]',  # SSNs
        r'\b\d{2}/\d{2}/\d{4}\b': '[REDACTED]',  # Dates in MM/DD/YYYY format
        r'\b\d{5}\b': '[REDACTED]',  # Zip codes
        r'\b\d{3}-\d{3}-\d{4}\b': '[REDACTED]',  # Phone numbers
        r'\b\d{10}\b': '[REDACTED]',  # Phone numbers without dashes
        r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b': '[REDACTED]',  # Emails
        r'\b\d{1,4}-\d{1,4}-\d{1,4}\b': '[REDACTED]',  # Medical record numbers
        r'\b(?:[A-Fa-f0-9]{2}:){5}[A-Fa-f0-9]{2}\b': '[REDACTED]',  # MAC addresses
        r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b': '[REDACTED]',  # IPv4 addresses
        r'\b(?:[0-9a-fA-F]{4}::?){2,7}\b': '[REDACTED]',  # IPv6 addresses
        r'\b(?:\d{4}[- ]){3}\d{4}\b': '[REDACTED]',  # Credit card numbers
        r'\b[1-9]\d{2}-\d{2}-\d{4}\b': '[REDACTED]',  # Alternate SSNs
        r'\b[A-Z]{3}-\d{3}-\d{3}\b': '[REDACTED]',  # License plate numbers
        r'\b\d{9}\b': '[REDACTED]',  # Generic ID numbers
        r'\b\d{4}\b': '[REDACTED]',  # Year
    }
    for pattern, replacement in patterns.items():
        deidentified_text = re.sub(pattern, replacement, deidentified_text, flags=re.IGNORECASE)
    return deidentified_text

def chunk_text(text, chunk_size=512):
    chunks = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))
        chunk = text[start_idx:end_idx]
        if end_idx < len(text) and not re.match(r"\b\w+\b$", chunk):
            chunk += " " + text[end_idx]
            end_idx += 1
        chunks.append(chunk)
        start_idx = end_idx
    return chunks

def generate_embeddings(text, model, tokenizer):
    chunks = chunk_text(text)
    embeddings = []
    for chunk in chunks:
        encoded_input = tokenizer(chunk, return_tensors="pt", padding="max_length", truncation=True)
        encoded_input = encoded_input.to(device)
        with torch.no_grad():
            output = model(**encoded_input)
            last_hidden_state = output.last_hidden_state
            chunk_embedding = torch.mean(last_hidden_state, dim=1)
            embeddings.append(chunk_embedding.cpu().detach().numpy())
    if embeddings:
        return np.mean(np.concatenate(embeddings), axis=0)
    else:
        return np.array([])

def extract_medical_entities(text):
    doc = nlp(text)
    medical_entities = " ".join([ent.text for ent in doc.ents])
    return medical_entities if medical_entities else "[EMPTY]"

def decode_base64_column(column):
    def decode_string(encoded_string):
        missing_padding = len(encoded_string) % 4
        if missing_padding != 0:
            encoded_string += '=' * (4 - missing_padding)
        return base64.b64decode(encoded_string).decode('utf-8')

    return column.apply(decode_string)

# Load the CSV file
file_path = '/oncmloutput/cqlplanfile.csv'
df = pd.read_csv(file_path, encoding='cp1252', skip_blank_lines=False, keep_default_na=False, na_filter=False)
df['Notes'] = decode_base64_column(df['Notes'])
# Initialize lists to store embeddings
note_embeddings = []
med_embeddings = []

def process_note(note):
    # Deidentify the note
    deidentified_note = deidentify_text(note)
    # Generate embeddings for the deidentified note
    note_embedding = generate_embeddings(deidentified_note, biobert_model, biobert_tokenizer)
    # Extract medical entities using med7
    medical_entities = extract_medical_entities(deidentified_note)
    # Generate embeddings for the extracted medical entities
    med_embedding = generate_embeddings(medical_entities, biobert_model, biobert_tokenizer)
    return note_embedding, med_embedding

# Process notes in parallel
with ThreadPoolExecutor(max_workers=8) as executor:
    results = executor.map(process_note, df['Notes'].astype(str).replace(r'^\s*$', "[EMPTY]", regex=True).fillna('[EMPTY]'))
    for note_embedding, med_embedding in results:
        note_embeddings.append(note_embedding.tolist() if note_embedding.size > 0 else [])
        med_embeddings.append(med_embedding.tolist() if med_embedding.size > 0 else [])

# Convert lists back to numpy arrays for projection
note_embeddings_np = np.array(note_embeddings)
med_embeddings_np = np.array(med_embeddings)

# Apply GaussianRandomProjection to both sets of embeddings
transformer = GaussianRandomProjection(n_components=768, random_state=42)
projected_note_embeddings = transformer.fit_transform(note_embeddings_np)
projected_med_embeddings = transformer.fit_transform(med_embeddings_np)

# Add the projected embeddings back to the DataFrame
df['Projected_Note_Embeddings'] = [list(embedding) for embedding in projected_note_embeddings]
df['Projected_Med_Embeddings'] = [list(embedding) for embedding in projected_med_embeddings]
df = df.drop(columns=['Notes'])

# Save the updated DataFrame to a new CSV and PKL file
output_file_path = '/oncmloutput/cqlplanfile_with_projected_embeddings.pkl'
df.reset_index(drop=True, inplace=True)
df.to_pickle(output_file_path)
output_file_path2 = '/oncmloutput/cqlplanfile_with_projected_embeddings.csv'
df.to_csv(output_file_path2, index=False)

print(f"Processed file saved to {output_file_path}")
print(f"Processed file saved to {output_file_path2}")
