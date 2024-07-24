# RAGFeynman

RAGFeynman is a question-answering assistant that leverages Retrieval-Augmented Generation (RAG) with large language models (LLMs) like Gemma or TinyLlama. This application uses a variety of tools and libraries to provide accurate and efficient answers to user queries.

## Features

- **Question Answering**: Provides detailed answers to user queries.
- **Retrieval-Augmented Generation (RAG)**: Combines retrieval-based and generation-based methods to enhance response accuracy.
- **Language Models**: Utilizes Gemma or TinyLlama for generating responses.

## Technologies Used

- [HuggingFace](https://huggingface.co/) - For accessing pre-trained models and embeddings.
- [Langchain](https://www.langchain.com/) - For building language model pipelines.
- [Streamlit](https://streamlit.io/) - For creating the web interface.
- [Torch](https://pytorch.org/) - For deep learning model implementations.
- [Transformers](https://huggingface.co/transformers/) - For working with transformer models.
- [pymudf](https://github.com/pymudf/pymudf) - For working with multi-document formats.
- [HFembeddings](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification) - For utilizing HuggingFace embeddings.
- [FAISS](https://github.com/facebookresearch/faiss) - For efficient similarity search.

## Installation

To run the RAGFeynman application locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Yadukrishnan1/RAGFeynman.git
    cd RAGFeynman
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set your HuggingFace token:
    ```bash
    export HF_TOKEN=<your_huggingface_token>  # On Windows use `set HF_TOKEN=<your_huggingface_token>`
    ```

## Usage

To start the Streamlit application:

```bash
streamlit run app.py
```


## How It Works

- **User Input**: The user enters a query into the web interface.

- **Retrieval**: Relevant documents or passages are retrieved using FAISS and HuggingFace embeddings.

- **Augmentation**: The retrieved information is augmented with the capabilities of Gemma or TinyLlama language models.

- **Generation**: A detailed and accurate answer is generated and displayed to the user.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
