# -*- coding: utf-8 -*-
import tkinter as tk
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from nltk.tokenize import word_tokenize


window = tk.Tk()
window.title('Clozer')
window.geometry('300x180')
window.configure(background='white')

def calculate_bmi_number():
    text = height_entry.get()
    ans = weight_entry.get()

    t = text.count('_')
    text = text.replace('_','',t-1)
    ans = ans.replace('A','',1)
    ans = ans.replace('B','',1)
    ans = ans.replace('C','',1)
    ans = ans.replace('D','',1)
    ans = ans.replace('(','')
    ans = ans.replace(')','')
    candidates = word_tokenize(ans)

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    tokenized_text = tokenizer.tokenize(text)

    masked_index = tokenized_text.index('_')
    tokenized_text[masked_index] = '[MASK]'


    candidates_ids = tokenizer.convert_tokens_to_ids(candidates)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [0] * len(tokenized_text)

    tokens_tensor = torch.Tensor([indexed_tokens])
    segments_tensors = torch.Tensor([segments_ids])

    language_model = BertForMaskedLM.from_pretrained('bert-large-uncased')
    language_model.eval()

    predictions = language_model(tokens_tensor, segments_tensors)
    predictions_candidates = predictions[0, masked_index, candidates_ids]
    answer_idx = torch._C.argmax(predictions_candidates).item()

    result = f"答案是:'{candidates[answer_idx]}'"

    result_label.configure(text=result)


header_label = tk.Label(window, text='題目填空部分請使用_ 答案須包含(A)(B)(C)(D)')
header_label.pack()

height_frame = tk.Frame(window)
height_frame.pack(side=tk.TOP)
height_label = tk.Label(height_frame, text='題目')
height_label.pack(side=tk.LEFT)
height_entry = tk.Entry(height_frame)
height_entry.pack(side=tk.LEFT)

weight_frame = tk.Frame(window)
weight_frame.pack(side=tk.TOP)
weight_label = tk.Label(weight_frame, text='答案')
weight_label.pack(side=tk.LEFT)
weight_entry = tk.Entry(weight_frame)
weight_entry.pack(side=tk.LEFT)

result_label = tk.Label(window)
result_label.pack()

calculate_btn = tk.Button(window, text='解題', command=calculate_bmi_number)
calculate_btn.pack()

window.mainloop()