import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

def _register_embedding_list_hook(model, embeddings_list):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    embedding_layer = model.get_input_embeddings()
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

def _register_embedding_gradient_hooks(model, embeddings_gradients):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0])
    embedding_layer = model.get_input_embeddings()
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook

def saliency_map(model, input_ids, input_mask):
    torch.enable_grad()
    model.eval()
    embeddings_list = []
    handle = _register_embedding_list_hook(model, embeddings_list)
    embeddings_gradients = []
    hook = _register_embedding_gradient_hooks(model, embeddings_gradients)

    model.zero_grad()
    A = model(input_ids, attention_mask=input_mask)
    pred_label_ids = np.argmax(A.logits[0].detach().numpy())
    A.logits[0][pred_label_ids].backward()
    handle.remove()
    hook.remove()

    saliency_grad = embeddings_gradients[0].detach().cpu().numpy()        
    saliency_grad = np.sum(saliency_grad[0] * embeddings_list[0], axis=1)
    norm = np.linalg.norm(saliency_grad, ord=1)
    saliency_grad = [e / norm for e in saliency_grad] 
    
    return saliency_grad

def get_num_top_tokens():
    global num_top_tokens
    num_top_tokens = st.slider('Number of top tokens', 10, 100, 35)

def get_top_indices(salience_scores, n):
    indexed_scores = [(index, float(score)) for index, score in enumerate(salience_scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in indexed_scores[:n]]

    return top_indices

def highlight_top_words(tokens, top_indices):
    # "[CLS]" and "[SEP]" are special tokens associated with models that use BERT. They may help with salience. Uncomment below line to remove them from the highlighted text.

    # tokens = [token for token in tokens if token not in ('[CLS]', '[SEP]')]

    highlighted_text = []

    for index, word in enumerate(tokens):
        if index in top_indices:
            highlighted_text.append(f'<span style="color: #ff5959;">{word}</span>')
        else:
            highlighted_text.append(word)

    return ' '.join(highlighted_text)

def run_ml_model(text):
    tokenizer = AutoTokenizer.from_pretrained("dlentr/lie_detection_distilbert")
    model = AutoModelForSequenceClassification.from_pretrained("dlentr/lie_detection_distilbert")
    inputs = tokenizer.encode(text, padding=True, truncation=True, return_tensors='pt')
    # input_ids = inputs[0].tolist()
    outputs = model(inputs)
    logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1)
    probability_class_0 = probabilities[0, 0].item()
    probability_class_1 = probabilities[0, 1].item()
    predicted_label = torch.argmax(probabilities, dim=1).item()
    
    if predicted_label == 0:
        result = 'Truth'
        confidence = round(probability_class_0*100,2)
    else:
        result = 'Lie'
        confidence = round(probability_class_1*100,2) 
 
    # def query(payload):
    #     response = requests.post(API_URL, headers=headers, json=payload)
    #     return response.json()

    # API_URL = "https://api-inference.huggingface.co/models/dlentr/lie_detection_distilbert"
    # headers = {"Authorization": f"Bearer {auth_token}"}

    # output = query({
    #     "inputs": {
    #     "text": text,
    #     "truncation": True,
    #     "padding": True
    # }
    # })

    # best_class = max(output, key=lambda x: x['score'])
    # best_label = best_class['label']
    # confidence = best_class['score']
    # prediction = 1 if best_label == 'POSITIVE' else 0

    # if prediction == 0:
    #     result = 'Truth'
    #     confidence = round(confidence*100,2)
    # else:
    #     result = 'Lie'
    #     confidence = round(confidence*100,2) 
    
    # Salience
    # tokenizer = AutoTokenizer.from_pretrained("dlentr/lie_detection_distilbert")
    # model = AutoModelForSequenceClassification.from_pretrained("dlentr/lie_detection_distilbert")
    # inputs = tokenizer.encode(text, padding=True, truncation=True, return_tensors='pt')
    salience_tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    tokens = tokenizer(text)
    input_ids = torch.tensor([tokens['input_ids']], dtype=torch.long) #perhaps has to be filled to size of tensor
    attention_ids = torch.tensor([tokens['attention_mask']], dtype=torch.long)

    saliency_scores = saliency_map(model, input_ids, attention_ids)

    top_indices = get_top_indices(saliency_scores, num_top_tokens)
    highlighted_text = highlight_top_words(salience_tokens, top_indices)
    saliency_df = create_saliency_table(saliency_scores, salience_tokens, top_indices)

    return result, confidence, highlighted_text, saliency_df

def create_saliency_table(saliency_scores, salience_tokens, indices):
    saliency_data = {'Token': [], 'Saliency Score': []}

    for index in indices:
        token = salience_tokens[index]
        score = saliency_scores[index]

        saliency_data['Token'].append(token)
        saliency_data['Saliency Score'].append(score)

    saliency_df = pd.DataFrame(saliency_data)

    return saliency_df

# Steamlit functions
def home_page():
    st.title("Defense argument classifier")
    st.write("Where AI analyses legal defense arguments to uncover the facts.")

def text_input_page():
    st.title("")
    st.write("Enter a defense argument in the textfield below. Alternatively select one from the provided list.")

    data = pd.read_csv("data.csv")

    selected_row = st.selectbox("Select argument:", data["ID"])
    if selected_row:
        selected_data = data[data["ID"] == selected_row]
        text = st.text_area("Input text:", selected_data.iloc[0]["text"], height=250)
    else:
        text = st.text_area("Input text", height=250)

    get_num_top_tokens()

    if st.button("Detect", key="submit_button"):
        result, confidence, highlighted_text, saliency_df = run_ml_model(text)
        
        st.write("Result:", result)
        st.write("Confidence:", confidence)

        st.markdown(highlighted_text, unsafe_allow_html=True)
        st.write('\n')
        col1, col2, col3 = st.columns([3,3,3])
        with col1:
            st.write('')
        with col2:
            st.dataframe(saliency_df)
        with col3:
            st.write('')


num_top_tokens = 10

home_page()
text_input_page()
