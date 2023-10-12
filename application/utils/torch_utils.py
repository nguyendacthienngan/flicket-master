# Here we want to do three things:

# Load the model
# Preprocess the image and convert it to a torch tensor
# Do the prediction

# https://www.python-engineer.com/posts/pytorch-model-deployment-with-flask/
import io
import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F
MAX_LEN = 512
PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'

model = XLNetForSequenceClassification.from_pretrained(
    PRE_TRAINED_MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load(
    r'C:\Users\Administrator\Documents\ERP\flicket-master\application\utils\xlnet_model.bin', map_location=torch.device('cpu')))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = model.to("cpu")
tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


def predict_sentiment(text):
    review_text = text

    encoded_review = tokenizer.encode_plus(
        review_text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=False,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = pad_sequences(
        encoded_review['input_ids'], maxlen=MAX_LEN, dtype=torch.Tensor, truncating="post", padding="post")
    input_ids = input_ids.astype(dtype='int64')
    input_ids = torch.tensor(input_ids)

    attention_mask = pad_sequences(
        encoded_review['attention_mask'], maxlen=MAX_LEN, dtype=torch.Tensor, truncating="post", padding="post")
    attention_mask = attention_mask.astype(dtype='int64')
    attention_mask = torch.tensor(attention_mask)

    input_ids = input_ids.reshape(1, 512).to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    outputs = outputs[0][0].cpu().detach()

    # Probabilities: [positive score, negative score]
    probs = F.softmax(outputs, dim=-1).cpu().detach().numpy().tolist()
    _, prediction = torch.max(outputs, dim=-1)
    class_names = ['negative', 'positive']
    print("Positive score:", probs[1])
    print("Negative score:", probs[0])
    print(f'Review text: {review_text}')
    print(f'Sentiment  : {class_names[prediction]}')
    return class_names[prediction]
