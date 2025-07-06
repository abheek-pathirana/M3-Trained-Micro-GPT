#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 07:05:39 2025

@author: abheekpathirana
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("SML-3.4m_Total_5_70e_70")
tokenizer = GPT2Tokenizer.from_pretrained("SML-3.4m_Total_5_70e_70")

print(model)

import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
model.eval()

prompt = "paris"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=100,
        do_sample=True,        # enables randomness
        top_k=50,              # top-k sampling
        top_p=0.95,            # nucleus sampling
        temperature=0.7,       # randomness factor
        pad_token_id=tokenizer.eos_token_id  # avoids warning
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print('*********promt**********')
print(prompt)
print("🧠 Generated Output:\n")
print(generated_text)