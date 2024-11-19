# src/test_qa_evaluation.py

import pytest
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

class TestAcquiredQA:
    @pytest.fixture
    def qa_system(self):
        embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        llm = OllamaLLM(model="llama3.2")
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(),
            return_source_documents=True
        )

    @pytest.fixture
    def test_cases(self):
        return pd.read_csv('src/acquired-qa-evaluation.csv')

    def test_airbnb_ipo_question(self, qa_system, test_cases):
        # Get test case from CSV
        test_row = test_cases[test_cases['question'].str.contains('Airbnb', case=False)].iloc[0]
        query = test_row['question']
        
        # Get response from QA system
        response = qa_system.invoke({"query": query})
        
        # Assert expected information is present
        expected_facts = [
            "December 10, 2020",
            "$68",
            "$3.5 billion"
        ]
        
        for fact in expected_facts:
            assert fact in response['result'], f"Expected {fact} in response"
        
        # Verify source documents
        assert len(response['source_documents']) > 0, "Expected source documents in response"