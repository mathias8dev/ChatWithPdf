# chatWithPDF
This project allows you to upload a PDF document and ask questions about its content. It uses langchain, openapi ai model and  Facebook Ai Similarity Search(FAISS) library to process the text in the PDF and provide answers to questions pertaining the document.


## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/mathias8dev/ChatWithPdf
   cd into your directory/ open with vscode
   ```
2. Create a Virtual Environment:
    ```shell
    python -m venv env
    ```
3. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```
4. Create .env file

   ```shell
   cp .env.example .env
   ```

5. Create OpenAI API Key and add it to your .env file:
   [openai](https://platform.openai.com/)
   
6. Run the application:

   ```shell
   python FaqApp.py
   ```

## Next Steps
1. Add support for multiple file formats
2. Implement Document Indexing techniques by use of libraries such as  Elasticsearch or Apache Solr 
3. Enhance question answering capabilities: Explore advanced question answering techniques, such as using transformer models like BERT or    GPT, to improve the accuracy and comprehension of the system.
4. Use a model that supports multiple languages, most notably some BERT models do support this.
