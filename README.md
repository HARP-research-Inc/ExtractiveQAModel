# ExtractiveQAModel
This project implements an Extractive Question Answering (QA) system using Hugging Face's transformers library. The model takes a natural language question and a passage of text (context), then returns a direct answer extracted from that context — just like SQuAD-style QA.

Features:
- Load context from any text file
- Ask multiple questions interactively
- Confidence scores shown as percentages
- Uses pre-trained Hugging Face model (no training required)
- Easy to modify or extend (e.g., for web, batch input, or fine-tuning)

General Notes:
- Results are based off on how much data and context is given in the context.txt file.
- If you are having installation problems, make sure to have all the python packages and streamlit installed in you terminal/miniconda shell.

How to Install:
 1. Clone the repository
```bash
git clone https://github.com/your-username/extractive-qa-bert.git
cd extractive-qa-bert

