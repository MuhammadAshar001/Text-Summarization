# Text Summarization

This project develops a text summarization system to generate concise summaries from lengthy articles, blogs, or news using the CNN/Daily Mail dataset. The goal is to create a model that performs both extractive and abstractive summarization for efficient and coherent summary generation.

## 📝 Overview

The `TASK06.ipynb` Jupyter Notebook implements the following steps:

- **Data Preprocessing**: Cleans and preprocesses textual data from the CNN/Daily Mail dataset for summarization tasks.
- **Extractive Summarization**: Uses spaCy to select key sentences based on relevance and importance.
- **Abstractive Summarization**: Leverages pre-trained models like BERT or GPT from HuggingFace's transformers for generating human-like summaries.
- **Model Fine-Tuning**: Fine-tunes the abstractive model to improve summary quality and coherence.
- **Evaluation**: Tests the model on real-world articles and evaluates summaries for coherence, relevance, and conciseness using metrics like ROUGE.

## 📊 Dataset

- **Source**: [CNN/Daily Mail Dataset](https://huggingface.co/datasets/cnn_dailymail) from the HuggingFace Datasets library.
- **Description**: Contains news articles from CNN and Daily Mail paired with their human-written summaries, designed for text summarization tasks.

## 🛠️ Requirements

To run the notebook, install the following Python libraries:

```bash
pip install pandas numpy spacy transformers datasets torch scikit-learn rouge-score
```
### Setup for spaCy:
```bash
python -m spacy download en_core_web_sm
```

## How to Run
1. Clone the Repository (if applicable, or skip to step 2):
``` bash
git clone https://github.com/your-username/text-summarization.git
cd text-summarization
```
2. Install Required Libraries:
``` bash
pip install pandas numpy spacy transformers datasets torch scikit-learn rouge-score
python -m spacy download en_core_web_sm
```
3. Run the Notebook:
- Ensure the CNN/Daily Mail dataset is accessible (automatically downloadable via HuggingFace's datasets library).
- Open TASK06.ipynb in Jupyter Notebook or JupyterLab and run all cells.

## Final Results
Abstractive Summarization (BERT/GPT-based):
- ROUGE-1: 0.0613
- ROUGE-2: 0.0207
- ROUGE-L: 0.0409
Coherence: Abstractive summaries show modest performance, likely due to limited fine-tuning or dataset complexity. Further fine-tuning or advanced models (e.g., BART, T5) could improve results.

## Contributing
Contributions are welcome! Fork the repo and submit a pull request for improvements, additional models (e.g., T5, BART), or enhanced preprocessing techniques.

## License
This project is open-source and available under the MIT License.
