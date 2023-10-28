import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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
    num_top_tokens = st.slider('Number of top words', 10, 100, 20)

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
            highlighted_text.append(f'<span style="color: red;">{word}</span>')
        else:
            highlighted_text.append(word)

    return ' '.join(highlighted_text)

def run_ml_model(text):
    tokenizer = AutoTokenizer.from_pretrained("dlentr/lie_detection_distilbert", token=auth_token)
    model = AutoModelForSequenceClassification.from_pretrained("dlentr/lie_detection_distilbert", token=auth_token)
    inputs = tokenizer.encode(text, padding=True, truncation=True, return_tensors='pt')
    # input_ids = inputs[0].tolist()
    outputs = model(inputs)
    logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1)
    probability_class_0 = probabilities[0, 0].item()
    probability_class_1 = probabilities[0, 1].item()
    predicted_label = torch.argmax(probabilities, dim=1).item()
    
    if predicted_label == 0:
        result = 'True'
        confidence = round(probability_class_0*100,2)
    else:
        result = 'False'
        confidence = round(probability_class_1*100,2) 
    
    # Salience
    salience_tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    tokens = tokenizer(text)
    input_ids = torch.tensor([tokens['input_ids']], dtype=torch.long) #perhaps has to be filled to size of tensor
    attention_ids = torch.tensor([tokens['attention_mask']], dtype=torch.long)

    saliency_scores = saliency_map(model, input_ids, attention_ids)

    top_indices = get_top_indices(saliency_scores, num_top_tokens)
    highlighted_text = highlight_top_words(salience_tokens, top_indices)
    # highlighted_text = ""

    return result, confidence, highlighted_text

# Steamlit functions
def home_page():
    st.title("Lie Detection")
    st.write("Welcome to Lie Detection: Where AI analyses legal defenses to uncover the facts.")

def text_input_page():
    st.title("")
    st.write("Enter a defense argument without putting it in quotation marks and click Detect.")
    text = st.text_area("Input text", height=250)

    get_num_top_tokens()

    if st.button("Detect", key="submit_button"):
        result, confidence, highlighted_text = run_ml_model(text)
        
        st.write("Result:", result)
        st.write("Confidence:", confidence)

        st.markdown(highlighted_text, unsafe_allow_html=True)


auth_token = "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"
num_top_tokens = 10

home_page()
text_input_page()
