<<<<<<< HEAD
# Medical Assistant Chatbot

A Retrieval-Augmented Generation (RAG) based medical assistant chatbot that provides educational medical information using a comprehensive dataset of 16,407 medical Q&A pairs.

## 🏥 Features

- **RAG System**: Uses FAISS for efficient similarity search across medical knowledge base
- **Medical Dataset**: Contains 16,407 cleaned medical Q&A pairs from MedQuad dataset
- **Web Interface**: Beautiful Streamlit chat interface with dark theme
- **Safety First**: Always includes medical disclaimers and safety warnings
- **Real-time Responses**: Instant answers to medical questions
- **Educational Focus**: Designed for educational purposes, not medical diagnosis

## 📋 Prerequisites

- Python 3.10+
- macOS (tested on MacBook)
- Virtual environment (recommended)

## 🚀 Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd medical-assistant-project
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install streamlit transformers torch sentence-transformers faiss-cpu pandas numpy plotly
   ```

## 📁 Project Structure

```
medical-assistant-project/
├── app.py                    # Main Streamlit web application
├── rag_system.py            # RAG system with FAISS and embeddings
├── prepare_data.py          # Data preparation script
├── medquad_complete.csv     # Complete medical dataset (16,407 Q&A pairs)
├── medquad_complete.json    # JSON version of the dataset
├── venv/                    # Virtual environment
└── README.md               # This file
```

## 🎯 Usage

### Starting the Application

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Access the application:**
   - Local URL: http://localhost:8501
   - Network URL: http://10.173.181.149:8501

### Using the Chatbot

1. **Open your browser** and navigate to the provided URL
2. **Type medical questions** in the chat input at the bottom
3. **Get instant responses** with relevant medical information
4. **View chat history** of your conversation

### Example Questions

- "What are the symptoms of heart attack?"
- "How to treat diabetes?"
- "What causes high blood pressure?"
- "Symptoms of COVID-19"
- "Treatment for migraine"

## 🔧 Technical Details

### RAG System Components

- **Embedding Model**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **Vector Database**: FAISS for fast similarity search
- **Knowledge Base**: 16,407 medical Q&A pairs from MedQuad dataset
- **Web Framework**: Streamlit for the user interface

### Data Processing

The system uses the MedQuad dataset which contains:
- Medical questions and answers
- Cleaned and preprocessed text
- Categorized medical information
- Educational medical content

### Safety Features

- **Medical Disclaimers**: Every response includes safety warnings
- **Educational Purpose**: Clearly stated for educational use only
- **Professional Consultation**: Always recommends consulting healthcare professionals
- **No Medical Advice**: System provides information, not diagnosis

## 🛠️ Customization

### Adding New Medical Data

1. **Prepare your data** in CSV format with `Question` and `Answer` columns
2. **Update the data loading** in `rag_system.py`
3. **Re-run the embedding generation** process

### Modifying the Model

1. **Change embedding model** in `rag_system.py`:
   ```python
   embedding_model = SentenceTransformer('your-model-name')
   ```

2. **Adjust search parameters**:
   ```python
   def retrieve_relevant_context(query, top_k=5):  # Change top_k
   ```

## 📊 Performance

- **Dataset Size**: 16,407 medical Q&A pairs
- **Embedding Dimension**: 384 (all-MiniLM-L6-v2)
- **Search Speed**: Fast FAISS similarity search
- **Response Time**: Near-instantaneous responses

## ⚠️ Important Notes

### Medical Disclaimer

This system is designed for **educational purposes only**. It should not be used for:
- Medical diagnosis
- Treatment decisions
- Emergency medical situations
- Professional medical advice

**Always consult qualified healthcare professionals for medical advice.**

### Limitations

- Responses are based on the training dataset
- May not include the most recent medical information
- Not a substitute for professional medical consultation
- Educational content only

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   streamlit run app.py --server.port 8502
   ```

2. **Memory issues:**
   - The system uses CPU-based models suitable for MacBooks
   - No GPU required

3. **Model loading errors:**
   - Ensure internet connection for model downloads
   - Check virtual environment activation

### Performance Tips

- **First run**: May take longer to download models
- **Subsequent runs**: Faster startup with cached models
- **Browser**: Use modern browsers for best experience

## 📈 Future Enhancements

Potential improvements:
- Add more medical datasets
- Implement conversation memory
- Add medical image analysis
- Integrate with medical APIs
- Add multilingual support
- Implement user feedback system

## 🤝 Contributing

This is an educational project. For medical applications, please:
1. Consult medical professionals
2. Follow healthcare regulations
3. Ensure proper medical oversight
4. Validate all medical information

## 📄 License

This project is for educational purposes. Please ensure compliance with:
- Medical information regulations
- Data privacy laws
- Healthcare guidelines
- Professional medical standards

## 🆘 Support

For technical issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure virtual environment is activated
4. Check system requirements

For medical questions, please consult qualified healthcare professionals.

---

**🏥 Remember: This is an educational tool, not a medical device. Always consult healthcare professionals for medical advice.** 
=======
# Medical-Assistant
>>>>>>> fa549fa8d4c6793cf3f4075d1288d2e24a9cfe61
