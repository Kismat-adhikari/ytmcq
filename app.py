from flask import Flask, render_template, request, jsonify
import os
import time
import requests
import yt_dlp
from pathlib import Path
import threading
import uuid
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Store transcription results in memory (use Redis/database in production)
transcription_results = {}

# Your API keys (loaded from .env)
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def cleanup_file(file_path):
    """Remove temporary audio file"""
    try:
        Path(file_path).unlink()
    except FileNotFoundError:
        pass

def download_audio(youtube_url, output_path):
    """Download audio from YouTube video"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path.replace('.mp3', '.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        # Add these options to fix 403 errors
        'extractor_args': {
            'youtube': {
                'skip': ['dash', 'hls']
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        },
        'cookiefile': None,
        'no_warnings': True,
        'ignoreerrors': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    return output_path

def upload_to_assemblyai(file_path, api_key):
    """Upload audio file to AssemblyAI"""
    headers = {'authorization': api_key}
    
    with open(file_path, 'rb') as f:
        response = requests.post(
            'https://api.assemblyai.com/v2/upload',
            headers=headers,
            files={'file': f}
        )
    
    if response.status_code == 200:
        return response.json()['upload_url']
    else:
        raise Exception(f"Upload failed: {response.status_code} - {response.text}")

def submit_transcription(audio_url, api_key, language_code=None):
    """Submit transcription request to AssemblyAI with language support"""
    headers = {
        'authorization': api_key,
        'content-type': 'application/json'
    }
    
    data = {
        'audio_url': audio_url,
        'language_detection': True,  # Enable automatic language detection
        'multichannel': False,
        'punctuate': True,
        'format_text': True,
    }
    
    # If specific language is provided, use it
    if language_code:
        data['language_code'] = language_code
        data['language_detection'] = False
    
    response = requests.post(
        'https://api.assemblyai.com/v2/transcript',
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json()['id']
    else:
        raise Exception(f"Transcription submission failed: {response.status_code} - {response.text}")

def poll_transcription(transcript_id, api_key, job_id):
    """Poll AssemblyAI for transcription completion"""
    headers = {'authorization': api_key}
    
    while True:
        response = requests.get(
            f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
            headers=headers
        )
        
        if response.status_code != 200:
            transcription_results[job_id] = {
                'status': 'error',
                'error': f"Polling failed: {response.status_code} - {response.text}"
            }
            return
        
        result = response.json()
        status = result['status']
        
        transcription_results[job_id] = {'status': status}
        
        if status == 'completed':
            # Generate MCQs and Flashcards using Groq
            transcription_results[job_id] = {'status': 'generating_content'}
            
            # Generate MCQs
            mcq_data = generate_mcqs_with_groq(
                result['text'], 
                result.get('language_detected', 'en')
            )
            
            # Generate Flashcards
            flashcard_data = generate_flashcards_with_groq(
                result['text'], 
                result.get('language_detected', 'en')
            )
            
            # Include all data in final result
            transcription_results[job_id] = {
                'status': 'completed',
                'transcript': result['text'],  # Keep transcript for internal use
                'language_detected': result.get('language_detected'),
                'language_confidence': result.get('confidence', 0),
                'audio_duration': result.get('audio_duration'),
                'mcqs': mcq_data.get('questions', []),
                'mcq_error': mcq_data.get('error'),
                'flashcards': flashcard_data.get('flashcards', []),
                'flashcard_error': flashcard_data.get('error')
            }
            return
        elif status == 'error':
            transcription_results[job_id] = {
                'status': 'error',
                'error': result.get('error', 'Unknown error')
            }
            return
        
        time.sleep(5)


def generate_mcqs_with_groq(transcript, language_detected="en"):
    """Generate relevant MCQs from transcript using Groq API"""
    
    # Content-focused prompts that analyze the actual transcript
    prompts = {
        "en": """You are an expert educator creating multiple choice questions based on the provided transcript content. Your goal is to create questions that test understanding of the key concepts, facts, and ideas presented in the material.

INSTRUCTIONS:
1. Analyze the transcript content carefully
2. Identify the main topics, key concepts, and important facts
3. Create 10-12 multiple choice questions that test understanding of this content
4. Questions should be educational and help students learn the material
5. Focus on comprehension, analysis, and application of the concepts presented
6. Make questions challenging but fair
7. Include a mix of difficulty levels (easy, medium, hard)
8. Only reference specific cultural contexts if they are actually mentioned in the transcript

QUESTION TYPES TO INCLUDE:
- Factual recall of key information
- Conceptual understanding questions
- Application of principles or concepts
- Analysis and synthesis questions
- Critical thinking questions

Return ONLY this JSON format:
{
  "questions": [
    {
      "question": "Clear, well-formed question based on transcript content?",
      "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
      "correct_answer": "A",
      "explanation": "Brief explanation of why this answer is correct",
      "subject": "Subject area based on transcript content",
      "difficulty": "easy/medium/hard",
      "topic": "Specific topic from transcript this question covers"
    }
  ]
}""",
        
        "hi": """आप एक विशेषज्ञ शिक्षक हैं जो दिए गए ट्रांसक्रिप्ट की सामग्री के आधार पर बहुविकल्पीय प्रश्न बना रहे हैं। आपका लक्ष्य ऐसे प्रश्न बनाना है जो सामग्री की मुख्य अवधारणाओं की समझ का परीक्षण करें।

निर्देश:
- ट्रांसक्रिप्ट की सामग्री का सावधानीपूर्वक विश्लेषण करें
- मुख्य विषयों और अवधारणाओं को पहचानें
- 10-12 बहुविकल्पीय प्रश्न बनाएं जो इस सामग्री की समझ का परीक्षण करें
- प्रश्न शैक्षणिक होने चाहिए और छात्रों को सीखने में मदद करने वाले हों

केवल JSON में उत्तर दें।""",
        
        "ne": """तपाईं एक विशेषज्ञ शिक्षक हुनुहुन्छ जसले दिइएको ट्रान्सक्रिप्टको सामग्री आधारमा बहुविकल्पीय प्रश्नहरू बनाइरहनुभएको छ। तपाईंको लक्ष्य यस्ता प्रश्नहरू बनाउनु हो जसले सामग्रीको मुख्य अवधारणाहरूको बुझाइको परीक्षण गर्छ।

निर्देशनहरू:
- ट्रान्सक्रिप्टको सामग्रीको सावधानीपूर्वक विश्लेषण गर्नुहोस्
- मुख्य विषयहरू र अवधारणाहरू पहिचान गर्नुहोस्
- 10-12 बहुविकल्पीय प्रश्नहरू बनाउनुहोस् जसले यस सामग्रीको बुझाइको परीक्षण गर्छ
- प्रश्नहरू शैक्षिक हुनुपर्छ र विद्यार्थीहरूलाई सिक्न मद्दत गर्नुपर्छ

JSON मा मात्र जवाफ दिनुहोस्।"""
    }
    
    prompt = prompts.get(language_detected, prompts["en"])
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Flexible strategies that adapt to content
    strategies = [
        {"max_tokens": 4000, "transcript_length": 3000, "questions": 12},
        {"max_tokens": 3500, "transcript_length": 2500, "questions": 10},
        {"max_tokens": 3000, "transcript_length": 2000, "questions": 9},
        {"max_tokens": 2500, "transcript_length": 1500, "questions": 8},
        {"max_tokens": 2000, "transcript_length": 1000, "questions": 7}
    ]
    
    for strategy in strategies:
        try:
            data = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"""You are an expert educator creating multiple choice questions based on transcript content. Your goal is to create {strategy['questions']} high-quality educational questions that test understanding of the material presented.

IMPORTANT GUIDELINES:
1. Focus on the actual content of the transcript
2. Create questions that help students understand the key concepts
3. Use appropriate difficulty levels (mix of easy, medium, hard)
4. Only reference specific contexts if they appear in the transcript
5. Make questions educational and meaningful
6. Avoid forcing unrelated cultural or geographical references
7. Focus on comprehension, analysis, and application

Return ONLY valid JSON format with no extra text."""
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nTranscript content to analyze:\n{transcript[:strategy['transcript_length']]}"
                    }
                ],
                "model": "llama3-70b-8192",
                "temperature": 0.3,
                "max_tokens": strategy["max_tokens"]
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code != 200:
                continue
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            print(f"MCQ Strategy: {strategy}")
            print(f"Raw response length: {len(content)}")
            print(f"Finish reason: {result.get('choices', [{}])[0].get('finish_reason')}")
            
            # Clean the response
            content = clean_json_response(content)
            
            # Try to parse
            try:
                mcq_data = json.loads(content)
                
                # Validate structure and ensure we have enough questions
                if validate_mcq_structure(mcq_data) and len(mcq_data.get('questions', [])) >= 5:
                    return mcq_data
                else:
                    print(f"Invalid MCQ structure or insufficient questions ({len(mcq_data.get('questions', []))}), trying next strategy")
                    continue
                    
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                fixed_content = fix_json_issues(content)
                if fixed_content:
                    try:
                        mcq_data = json.loads(fixed_content)
                        if validate_mcq_structure(mcq_data) and len(mcq_data.get('questions', [])) >= 5:
                            return mcq_data
                    except:
                        pass
                continue
                
        except Exception as e:
            print(f"MCQ Strategy failed: {e}")
            continue
    
    # Content-focused fallback questions
    return {
        "questions": [
            {
                "question": "Based on the content presented, which approach would be most effective for understanding this topic?",
                "options": ["A) Memorizing all details", "B) Understanding key concepts and their relationships", "C) Focusing only on specific examples", "D) Skipping complex parts"],
                "correct_answer": "B",
                "explanation": "Understanding key concepts and their relationships provides a solid foundation for learning any subject matter.",
                "subject": "General Learning",
                "difficulty": "easy",
                "topic": "Learning strategies"
            },
            {
                "question": "When analyzing complex information, what is the most important first step?",
                "options": ["A) Jumping to conclusions", "B) Identifying the main ideas and themes", "C) Memorizing every detail", "D) Comparing with unrelated topics"],
                "correct_answer": "B",
                "explanation": "Identifying main ideas and themes helps organize information and provides a framework for deeper understanding.",
                "subject": "Critical Thinking",
                "difficulty": "medium",
                "topic": "Analysis techniques"
            }
        ],
        "error": "Unable to generate comprehensive questions from the provided transcript. The content might be too short or unclear. Please try with clearer, more substantial educational content."
    }


def generate_flashcards_with_groq(transcript, language_detected="en"):
    """Generate relevant flashcards from transcript using Groq API"""
    
    # Content-focused prompts that analyze the actual transcript
    prompts = {
        "en": """You are an expert educator creating study flashcards based on the provided transcript content. Your goal is to create flashcards that help students learn and remember the key concepts, facts, and ideas presented in the material.

INSTRUCTIONS:
1. Analyze the transcript content carefully
2. Identify important terms, concepts, definitions, processes, and facts
3. Create 12-15 flashcards that help students study this material effectively
4. Include a variety of flashcard types: definitions, explanations, examples, processes
5. Focus on the most important and useful information from the transcript
6. Make flashcards that would actually help someone learn the topic
7. Only reference specific contexts if they are mentioned in the transcript

FLASHCARD TYPES TO INCLUDE:
- Key term definitions
- Important concept explanations
- Process descriptions
- Factual information
- Examples and applications
- Cause and effect relationships

Return ONLY this JSON format:
{
  "flashcards": [
    {
      "front": "Question, term, or concept from the transcript",
      "back": "Detailed answer, definition, or explanation",
      "subject": "Subject area based on transcript content",
      "difficulty": "easy/medium/hard",
      "category": "Definition/Explanation/Process/Fact/Example",
      "topic": "Specific topic from transcript this flashcard covers"
    }
  ]
}""",
        
        "hi": """आप एक विशेषज्ञ शिक्षक हैं जो दिए गए ट्रांसक्रिप्ट की सामग्री के आधार पर अध्ययन फ्लैशकार्ड बना रहे हैं। आपका लक्ष्य ऐसे फ्लैशकार्ड बनाना है जो छात्रों को मुख्य अवधारणाओं को सीखने और याद रखने में मदद करें।

निर्देश:
- ट्रांसक्रिप्ट की सामग्री का सावधानीपूर्वक विश्लेषण करें
- महत्वपूर्ण शब्दों, अवधारणाओं और तथ्यों को पहचानें
- 12-15 फ्लैशकार्ड बनाएं जो इस सामग्री को प्रभावी रूप से सिखाने में मदद करें
- विभिन्न प्रकार के फ्लैशकार्ड शामिल करें: परिभाषाएं, व्याख्याएं, उदाहरण

केवल JSON में उत्तर दें।""",
        
        "ne": """तपाईं एक विशेषज्ञ शिक्षक हुनुहुन्छ जसले दिइएको ट्रान्सक्रिप्टको सामग्री आधारमा अध्ययन फ्ल्यासकार्डहरू बनाइरहनुभएको छ। तपाईंको लक्ष्य यस्ता फ्ल्यासकार्डहरू बनाउनु हो जसले विद्यार्थीहरूलाई मुख्य अवधारणाहरू सिक्न र सम्झन मद्दत गर्छ।

निर्देशनहरू:
- ट्रान्सक्रिप्टको सामग्रीको सावधानीपूर्वक विश्लेषण गर्नुहोस्
- महत्वपूर्ण शब्दहरू, अवधारणाहरू र तथ्यहरू पहिचान गर्नुहोस्
- 12-15 फ्ल्यासकार्डहरू बनाउनुहोस् जसले यस सामग्रीलाई प्रभावकारी रूपमा सिकाउन मद्दत गर्छ
- विभिन्न प्रकारका फ्ल्यासकार्डहरू समावेश गर्नुहोस्: परिभाषाहरू, व्याख्याहरू, उदाहरणहरू

JSON मा मात्र जवाफ दिनुहोस्।"""
    }
    
    prompt = prompts.get(language_detected, prompts["en"])
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Flexible strategies for flashcard generation
    strategies = [
        {"max_tokens": 3500, "transcript_length": 2500, "flashcards": 15},
        {"max_tokens": 3000, "transcript_length": 2000, "flashcards": 12},
        {"max_tokens": 2500, "transcript_length": 1500, "flashcards": 10},
        {"max_tokens": 2000, "transcript_length": 1000, "flashcards": 8}
    ]
    
    for strategy in strategies:
        try:
            data = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"""You are an expert educator creating study flashcards based on transcript content. Your goal is to create {strategy['flashcards']} high-quality flashcards that help students learn the material presented.

IMPORTANT GUIDELINES:
1. Focus on the actual content of the transcript
2. Create flashcards that help students understand key concepts
3. Include various types: definitions, explanations, processes, facts
4. Only reference specific contexts if they appear in the transcript
5. Make flashcards educational and practical for studying
6. Avoid forcing unrelated cultural or geographical references
7. Focus on the most important information from the transcript

Return ONLY valid JSON format with no extra text."""
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nTranscript content to analyze:\n{transcript[:strategy['transcript_length']]}"
                    }
                ],
                "model": "llama3-70b-8192",
                "temperature": 0.2,
                "max_tokens": strategy["max_tokens"]
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code != 200:
                continue
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            print(f"Flashcard Strategy: {strategy}")
            print(f"Raw response length: {len(content)}")
            print(f"Finish reason: {result.get('choices', [{}])[0].get('finish_reason')}")
            
            # Clean the response
            content = clean_json_response(content)
            
            # Try to parse
            try:
                flashcard_data = json.loads(content)
                
                # Validate structure
                if validate_flashcard_structure(flashcard_data) and len(flashcard_data.get('flashcards', [])) >= 3:
                    return flashcard_data
                else:
                    print(f"Invalid flashcard structure or insufficient flashcards ({len(flashcard_data.get('flashcards', []))}), trying next strategy")
                    continue
                    
            except json.JSONDecodeError as e:
                print(f"Flashcard JSON parse error: {e}")
                fixed_content = fix_json_issues(content)
                if fixed_content:
                    try:
                        flashcard_data = json.loads(fixed_content)
                        if validate_flashcard_structure(flashcard_data) and len(flashcard_data.get('flashcards', [])) >= 3:
                            return flashcard_data
                    except:
                        pass
                continue
                
        except Exception as e:
            print(f"Flashcard Strategy failed: {e}")
            continue
    
    # Content-focused fallback flashcards
    return {
        "flashcards": [
            {
                "front": "What is the most effective approach to learning new information?",
                "back": "Active engagement with the material, connecting new concepts to existing knowledge, and regular review and practice. This approach helps build long-term understanding rather than temporary memorization.",
                "subject": "Learning Strategies",
                "difficulty": "medium",
                "category": "Explanation",
                "topic": "Effective learning methods"
            },
            {
                "front": "Why is understanding concepts better than memorizing facts?",
                "back": "Understanding concepts allows you to apply knowledge in different situations, solve new problems, and build connections between ideas. Memorization only provides temporary recall without deeper comprehension.",
                "subject": "Educational Psychology",
                "difficulty": "medium",
                "category": "Explanation",
                "topic": "Learning vs. memorization"
            },
            {
                "front": "What should you do when encountering difficult material?",
                "back": "Break it down into smaller parts, identify key concepts, look for patterns and relationships, ask questions, and connect it to what you already know. Seek help when needed and practice regularly.",
                "subject": "Study Skills",
                "difficulty": "easy",
                "category": "Process",
                "topic": "Handling difficult content"
            }
        ],
        "error": "Unable to generate comprehensive flashcards from the provided transcript. The content might be too short or unclear. Please try with clearer, more substantial educational content."
    }


def validate_flashcard_structure(flashcard_data):
    """Validate the flashcard data structure"""
    try:
        if not isinstance(flashcard_data, dict) or 'flashcards' not in flashcard_data:
            return False
        
        flashcards = flashcard_data['flashcards']
        if not isinstance(flashcards, list) or len(flashcards) == 0:
            return False
        
        for card in flashcards:
            if not isinstance(card, dict):
                return False
            
            required_keys = ['front', 'back']
            if not all(key in card for key in required_keys):
                return False
            
            if not isinstance(card['front'], str) or not isinstance(card['back'], str):
                return False
            
            if len(card['front'].strip()) == 0 or len(card['back'].strip()) == 0:
                return False
        
        return True
    except:
        return False


def clean_json_response(content):
    """Clean the JSON response from common formatting issues"""
    # Remove markdown code blocks
    if content.startswith('```json'):
        content = content.replace('```json', '').replace('```', '').strip()
    elif content.startswith('```'):
        content = content.replace('```', '').strip()
    
    # Remove any text before the first {
    start = content.find('{')
    if start > 0:
        content = content[start:]
    
    # Remove any text after the last }
    end = content.rfind('}')
    if end != -1:
        content = content[:end + 1]
    
    return content


def fix_json_issues(content):
    """Try to fix common JSON truncation issues"""
    try:
        # Count braces to see if we're missing closing ones
        open_braces = content.count('{')
        close_braces = content.count('}')
        open_brackets = content.count('[')
        close_brackets = content.count(']')
        
        # If we're missing closing braces/brackets, try to add them
        if open_braces > close_braces:
            # Try to complete the JSON structure
            missing_braces = open_braces - close_braces
            
            # If it looks like we're in the middle of a string, close it
            if content.count('"') % 2 == 1:
                content += '"'
            
            # Add missing closing brackets first
            if open_brackets > close_brackets:
                content += ']' * (open_brackets - close_brackets)
            
            # Add missing closing braces
            content += '}' * missing_braces
            
            return content
    except:
        pass
    
    return None


def validate_mcq_structure(mcq_data):
    """Validate the MCQ data structure"""
    try:
        if not isinstance(mcq_data, dict) or 'questions' not in mcq_data:
            return False
        
        questions = mcq_data['questions']
        if not isinstance(questions, list) or len(questions) == 0:
            return False
        
        for q in questions:
            if not isinstance(q, dict):
                return False
            
            required_keys = ['question', 'options', 'correct_answer']
            if not all(key in q for key in required_keys):
                return False
            
            if not isinstance(q['options'], list) or len(q['options']) != 4:
                return False
            
            if q['correct_answer'] not in ['A', 'B', 'C', 'D']:
                return False
        
        return True
    except:
        return False


def process_transcription(youtube_url, job_id, language_code=None):
    """Background task to process transcription"""
    audio_file = f"temp_audio_{job_id}.mp3"
    
    try:
        transcription_results[job_id] = {'status': 'downloading'}
        download_audio(youtube_url, audio_file)
        
        transcription_results[job_id] = {'status': 'uploading'}
        audio_url = upload_to_assemblyai(audio_file, ASSEMBLYAI_API_KEY)
        
        transcription_results[job_id] = {'status': 'submitting'}
        transcript_id = submit_transcription(audio_url, ASSEMBLYAI_API_KEY, language_code)
        
        transcription_results[job_id] = {'status': 'processing'}
        poll_transcription(transcript_id, ASSEMBLYAI_API_KEY, job_id)
        
    except Exception as e:
        transcription_results[job_id] = {
            'status': 'error',
            'error': str(e)
        }
    
    finally:
        cleanup_file(audio_file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    youtube_url = data.get('url')
    language_code = data.get('language')  # Optional language selection
    
    if not youtube_url:
        return jsonify({'error': 'YouTube URL is required'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Start background transcription process
    thread = threading.Thread(target=process_transcription, args=(youtube_url, job_id, language_code))
    thread.daemon = True
    thread.start()
    
    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>')
def status(job_id):
    result = transcription_results.get(job_id, {'status': 'not_found'})
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)