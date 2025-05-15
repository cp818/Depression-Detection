# Depression Detection

This project uses Deepgram's Nova-3 Speech-to-Text API to analyze speech patterns in real-time for potential biomarkers of depression. The system processes live audio, transcribes speech, and applies linguistic and sentiment analysis to assess depression risk.

## Features

- Real-time speech transcription using Deepgram Nova-3 API
- Live analysis of linguistic biomarkers associated with depression
- Depression risk scoring on a 0-100 scale
- Interactive web interface with visualization of results
- WebSocket-based streaming for low-latency processing

## Technology Stack

- Python FastAPI backend
- Deepgram Speech-to-Text API (Nova-3 model)
- NLTK for natural language processing
- scikit-learn for feature extraction
- WebSockets for real-time communication
- HTML/JavaScript frontend

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cp818/Depression-Detection.git
cd Depression-Detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Obtaining a Deepgram API key:
   - Sign up for a Deepgram account at https://console.deepgram.com/signup
   - Create a new API key in the Deepgram Console
   - You'll need this API key to use the application

### Using Your Deepgram API Key

This application requires a Deepgram API key to function. There are two ways to provide your API key:

**Option 1: Web Interface (Recommended)**
- When you open the web interface, you'll find an API key input field at the top
- Enter your Deepgram API key and click "Save API Key"
- The key will be securely stored in your browser's localStorage
- You only need to enter it once (unless you clear your browser data)

**Option 2: Environment File**
- For CLI usage or development, copy the `.env.example` file to `.env` and add your API key:
```bash
cp .env.example .env
# Edit .env and add: DEEPGRAM_API_KEY=your_key_here
```

## Usage

1. Start the server:
```bash
python app.py
```

2. Open your browser and navigate to http://localhost:8000

3. Click the microphone button and speak. The system will:
   - Transcribe your speech in real-time
   - Analyze the text for depression biomarkers
   - Display the depression risk score and analysis

## How It Works

The system analyzes several linguistic features that have been associated with depression in research:

1. **Sentiment Analysis**: Measures the emotional tone of speech
2. **Depression-Related Keywords**: Tracks usage of words associated with depression
3. **Self-Referential Language**: Measures frequency of first-person pronouns (indicator of self-focus)
4. **Speech Rate**: Analyzes pace of speech delivery
5. **Word Variety**: Measures lexical diversity and richness of language
6. **Pause Frequency**: Approximates speech hesitations and pauses

These features are weighted and combined to produce a depression risk score from 0-100.

## Limitations

- This tool is **NOT** a clinical diagnostic tool
- False positives and negatives are possible
- Speech patterns can be affected by many factors besides mental health
- Always consult a qualified mental health professional for diagnosis and treatment

## License

[MIT License](LICENSE)

## Disclaimer

This application is for educational and research purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.
