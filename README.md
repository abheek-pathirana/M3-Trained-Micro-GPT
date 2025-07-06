# M3-Trained-Micro-GPT
A 5.6M param GPT-2-style LLM pre-trained from scratch on an Apple M3 MacBook Air using 8.1M tokens and 70 epochs (6.41 petaFLOPs). Shows it’s possible to pre-train tiny LLMs on consumer hardware using PyTorch + FP32. Includes model, configs, training code, and samples.

model info:
            vocab_size=50257,
            n_positions=256,
            n_ctx=256,
            n_embd=96,
            n_layer=7,
            n_head=4


	•	Parameters: 5.6 million
	•	Tokens Used: 8.1 million (chunked to 256 individual tokens)
	•	Epochs: 70
	•	Compute Used: ~6.41 petaFLOPs (not accounting for )
	•	Time: ~18 hours
	•	Precision: Float32
	•	Device: Apple M3 GPU (Integrated 8core gpu)


