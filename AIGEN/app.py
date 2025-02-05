from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import openai
import requests
import os
from dotenv import load_dotenv
import json
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import io
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import time
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, 'app.log')
    
    handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Application startup')


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure AI Services
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')

# Validate required API keys
if not PEXELS_API_KEY:
    raise ValueError("PEXELS_API_KEY not found in environment variables")

# Initialize AI models
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ContentGenerator:
    def __init__(self, model_choice: str = 'gemini'):
        self.model_choice = model_choice.lower()
        if self.model_choice == 'gemini' and not GEMINI_API_KEY:
            raise ValueError("Gemini API key not configured")
        if self.model_choice == 'openai' and not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured")

    def generate_content(self, prompt: str) -> str:
        """
        Generates content using the specified AI model.
        
        Args:
            prompt: The prompt to send to the AI model
            
        Returns:
            str: Generated content
            
        Raises:
            Exception: If content generation fails
        """
        try:
            logger.debug(f"Attempting content generation with {self.model_choice}")
            if self.model_choice == 'gemini':
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                if not response.text:
                    raise ValueError("Empty response from Gemini")
                return response.text
                
            elif self.model_choice == 'openai':
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a professional content creator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                if not response.choices:
                    raise ValueError("Empty response from OpenAI")
                return response.choices[0].message.content
                
            else:
                raise ValueError(f"Unsupported model choice: {self.model_choice}")
                
        except Exception as e:
            logger.error(f"Detailed error for {self.model_choice}: {str(e)}")
            logger.error(f"Content generation error ({self.model_choice}): {str(e)}")
            raise Exception(f"Failed to generate content: {str(e)}")

class MetaPromptGenerator:
    def __init__(self, model_choice: str = 'gemini'):
        self.generator = ContentGenerator(model_choice)

    def generate_meta_prompt(self, platform: str, idea: str, content_type: str = "post") -> dict:
        if not platform or not idea:
            return {
                'success': False,
                'error': 'Platform and idea are required'
            }

        platform_guidelines = {
            'linkedin': {
                'tone': 'professional and insightful',
                'length': 'between 1200-1600 characters',
                'format': 'includes paragraph breaks and bullet points',
                'elements': 'professional insights, industry trends, data points'
            },
            'twitter': {
                'tone': 'concise and engaging',
                'length': 'maximum 280 characters',
                'format': 'includes hashtags',
                'elements': 'hooks, calls to action, trending topics'
            },
            'instagram': {
                'tone': 'visual and engaging',
                'length': 'between 800-1200 characters',
                'format': 'includes emojis and line breaks',
                'elements': 'storytelling, visual descriptions, hashtags'
            }
        }

        guidelines = platform_guidelines.get(platform.lower(), {
            'tone': 'balanced and clear',
            'length': 'appropriate for the platform',
            'format': 'well-structured',
            'elements': 'key messaging and engagement hooks'
        })

        meta_prompt = f"""
        Create a highly optimized prompt for generating {platform} content about: {idea}

        Follow this structured approach:

        1. Content Strategy:
        - Primary goal: Engage {platform} audience
        - Tone: {guidelines['tone']}
        - Length: {guidelines['length']}
        - Format: {guidelines['format']}
        
        2. Required Elements:
        - {guidelines['elements']}
        - Clear value proposition
        - Engagement triggers
        - Call to action
        """

        try:
            optimized_prompt = self.generator.generate_content(meta_prompt)
            if not optimized_prompt:
                raise ValueError("Failed to generate optimized prompt")

            final_content = self.generator.generate_content(optimized_prompt)
            if not final_content:
                raise ValueError("Failed to generate final content")

            return {
                'meta_prompt': optimized_prompt,
                'generated_content': final_content,
                'platform': platform,
                'original_idea': idea,
                'success': True
            }

        except Exception as e:
            logger.error(f"Meta prompt generation error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'platform': platform,
                'original_idea': idea
            }

    def validate_output(self, content: str, platform: str) -> dict:
        if not content or not platform:
            return {
                'success': False,
                'error': 'Content and platform are required'
            }

        validation_prompt = f"""
        Analyze this content for {platform} effectiveness:
        Content: {content}
        Evaluate:
        1. Platform alignment
        2. Message clarity
        3. Engagement potential
        """

        try:
            analysis = self.generator.generate_content(validation_prompt)
            if not analysis:
                raise ValueError("Failed to generate content analysis")

            return {
                'success': True,
                'analysis': analysis,
                'platform': platform
            }
        except Exception as e:
            logger.error(f"Content validation error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
        """
        Validates if the generated content meets platform requirements.
        
        Args:
            content: The generated content
            platform: The target platform
            
        Returns:
            dict: Validation results and suggestions
        """
        validation_prompt = f"""
        Analyze this content for {platform} effectiveness:

        Content: {content}

        Evaluate:
        1. Platform alignment
        2. Message clarity
        3. Engagement potential
        4. Call-to-action strength
        5. Overall optimization

        Provide specific improvement suggestions if needed.
        """

        try:
            analysis = self.generator.generate_content(validation_prompt)
            return {
                'success': True,
                'analysis': analysis,
                'platform': platform
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
class QAAgent:
    def __init__(self, model_choice: str = 'gemini'):
        self.generator = ContentGenerator(model_choice)
        self.max_retries = 3
        
    def validate_content(self, 
                        generated_content: str, 
                        original_style: str, 
                        topic: str) -> Tuple[bool, float, str]:
        prompt = f"""
        Analyze this content and compare it to the original style and topic:
        
        Content: {generated_content}
        Original Style: {original_style}
        Topic: {topic}
        
        Evaluate:
        1. Style match (0-100%)
        2. Topic relevance (0-100%)
        3. Overall quality
        
        Provide specific feedback for improvements if needed.
        """
        
        analysis = self.generator.generate_content(prompt)
        # Parse the analysis to extract confidence scores and feedback
        # This is a simplified version - you might want to add more sophisticated parsing
        confidence = 0.75  # Example confidence score
        needs_revision = confidence < 0.85
        return not needs_revision, confidence, analysis

class StrictStyleContentGenerator:
    def __init__(self, model_choice: str = 'gemini'):
        self.generator = ContentGenerator(model_choice)
        self.qa_agent = QAAgent(model_choice)
        
    def generate_style_matched_content(self, 
                                     topic: str, 
                                     style_text: str, 
                                     max_retries: int = 3) -> Dict[str, Any]:
        attempts = 0
        best_content = None
        best_confidence = 0
        
        while attempts < max_retries:
            prompt = f"""
            Write content about: {topic}
            
            Match this writing style exactly:
            {style_text}
            
            Ensure the content maintains:
            1. Same tone and voice
            2. Similar sentence structure
            3. Comparable vocabulary level
            4. Matching writing patterns
            5. Similar rhetorical devices
            """
            
            content = self.generator.generate_content(prompt)
            is_valid, confidence, feedback = self.qa_agent.validate_content(
                content, style_text, topic
            )
            
            if is_valid:
                return {
                    "content": content,
                    "confidence": confidence,
                    "attempts": attempts + 1
                }
            
            if confidence > best_confidence:
                best_content = content
                best_confidence = confidence
                
            attempts += 1
            time.sleep(1)  # Prevent rate limiting
        
        return {
            "content": best_content,
            "confidence": best_confidence,
            "attempts": attempts,
            "warning": "Maximum retries reached. Using best attempt."
        }

class StyleAnalyzer:
    def __init__(self, model_choice: str = 'gemini'):
        self.generator = ContentGenerator(model_choice)

    def analyze_style(self, text: str) -> str:
        prompt = f"""
        Analyze the writing style of this text and provide a detailed breakdown:
        
        Text: {text}
        
        Analyze:
        1. Tone and voice
        2. Sentence structure
        3. Vocabulary preferences
        4. Writing patterns
        5. Rhetorical devices used
        """
        
        return self.generator.generate_content(prompt)

class EngagementOptimizer:
    def __init__(self, model_choice: str = 'gemini'):
        self.generator = ContentGenerator(model_choice)
    
    def optimize_engagement(self, content: str, platform: str) -> str:
        prompt = f"""
        Optimize this content for {platform} engagement:
        
        Content: {content}
        
        Add:
        1. AIDA framework elements
        2. Platform-appropriate hooks
        3. Relevant CTAs
        4. Engagement triggers
        5. Platform-specific formatting
        
        Maintain the original message while maximizing engagement potential.
        """
        return self.generator.generate_content(prompt)

class MultimediaHandler:
    def __init__(self):
        self.pexels_api_key = PEXELS_API_KEY

    def get_media(self, query: str, media_type: str) -> Optional[str]:
        try:
            headers = {'Authorization': self.pexels_api_key}
            
            if media_type == 'image':
                response = requests.get(
                    f'https://api.pexels.com/v1/search?query={query}&per_page=1',
                    headers=headers,
                    timeout=10
                )
                if response.status_code == 200:
                    photos = response.json().get('photos', [])
                    if photos:
                        return photos[0]['src']['large']

            elif media_type == 'video':
                response = requests.get(
                    f'https://api.pexels.com/videos/search?query={query}&per_page=1',
                    headers=headers,
                    timeout=10
                )
                if response.status_code == 200:
                    videos = response.json().get('videos', [])
                    if videos and videos[0].get('video_files'):
                        return videos[0]['video_files'][0]['link']
                        
            return None
            
        except Exception as e:
            logger.error(f"Multimedia fetch error: {str(e)}")
            return None

class ContentManager:
    def __init__(self, model_choice: str = 'gemini'):
        self.generator = ContentGenerator(model_choice)
        self.multimedia = MultimediaHandler()

    def generate_platform_content(self, platform: str, topic: str, preferences: str = "") -> str:
        platform_guidelines = {
            'linkedin': (
                "Professional tone, industry insights, 1300 character limit, "
                "focus on business value and expertise"
            ),
            'twitter': (
                "Concise, engaging, 280 character limit, hashtags, "
                "conversational tone"
            ),
            'instagram': (
                "Visual-first approach, engaging story, relevant hashtags, "
                "emotional connection"
            )
        }

        prompt = f"""
        Create a {platform} post about: {topic}
        
        Platform guidelines:
        {platform_guidelines.get(platform.lower(), "Standard format")}
        
        Additional preferences:
        {preferences}
        
        Ensure the content:
        1. Follows platform best practices
        2. Includes appropriate engagement hooks
        3. Uses relevant calls-to-action
        4. Maintains proper tone for the platform
        """

        return self.generator.generate_content(prompt)

def extract_text_from_file(file) -> str:
    try:
        filename = secure_filename(file.filename)
        extension = filename.rsplit('.', 1)[1].lower()
        
        if extension == 'txt':
            return file.read().decode('utf-8')
        
        elif extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            return " ".join(page.extract_text() for page in pdf_reader.pages)
        
        elif extension == 'docx':
            doc = docx.Document(io.BytesIO(file.read()))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
            
        raise ValueError(f"Unsupported file type: {extension}")
        
    except Exception as e:
        logger.error(f"File extraction error: {str(e)}")
        raise

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        text_content = extract_text_from_file(file)
        
        # Get AI model choice from request
        model_choice = request.form.get('model', 'gemini')
        analyzer = StyleAnalyzer(model_choice)
        style_analysis = analyzer.analyze_style(text_content)
        
        return jsonify({
            'success': True,
            'style_analysis': style_analysis,
            'original_text': text_content
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required_fields = ['mode', 'platform', 'topic', 'model']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        model_choice = data.get('model', 'gemini')
        manager = ContentManager(model_choice)

        # Generate content based on mode
        if data['mode'] == 'strict_style' and data.get('style_text'):
            strict_generator = StrictStyleContentGenerator(model_choice)
            result = strict_generator.generate_style_matched_content(
                data['topic'],
                data['style_text']
            )
            content = result['content']
        else:
            content = manager.generate_platform_content(
                data['platform'],
                data['topic'],
                data.get('preferences', '')
            )

        # Optimize engagement if requested
        if data.get('optimize_engagement', False):
            optimizer = EngagementOptimizer(model_choice)
            content = optimizer.optimize_engagement(content, data['platform'])

        # Handle media if requested
        media_url = None
        if data.get('media_type') in ['image', 'video']:
            media_url = manager.multimedia.get_media(
                data['topic'],
                data['media_type']
            )

        return jsonify({
            'success': True,
            'content': content,
            'media_url': media_url
        })

    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/meta-prompt', methods=['POST'])
def generate_meta_prompt():
    try:
        # Log incoming request
        logger.debug(f"Request headers: {dict(request.headers)}")
        logger.debug(f"Request data: {request.get_data()}")
        
        # Parse JSON data with force=True to handle potential content-type issues
        data = request.get_json(force=True)
        logger.debug(f"Parsed JSON data: {data}")
        
        # Get required fields
        platform = data.get('platform')
        # Check both 'topic' and 'idea' fields since frontend might use either
        topic = data.get('topic') or data.get('idea')
        
        if not platform or not topic:
            raise ValueError(f'Platform and topic/idea are required fields. Received platform: {platform}, topic/idea: {topic}')

        # Get optional fields with defaults
        model_choice = data.get('model', 'gemini')
        content_type = data.get('content_type', 'post')
        
        # Generate meta prompt
        meta_generator = MetaPromptGenerator(model_choice)
        result = meta_generator.generate_meta_prompt(platform, topic, content_type)
        
        if not result.get('success'):
            logger.error(f"Meta prompt generation failed: {result.get('error')}")
            return jsonify(result), 500
            
        # Validate the generated content
        if result.get('generated_content'):
            validation = meta_generator.validate_output(result['generated_content'], platform)
            
            if not validation.get('success'):
                logger.error(f"Content validation failed: {validation.get('error')}")
                return jsonify({
                    'success': False,
                    'error': f"Validation failed: {validation.get('error')}"
                }), 500

        return jsonify(result)

    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in meta-prompt generation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
        app.logger.error(f"Error with request data: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400
        logger.error(f"Unexpected error in meta-prompt generation: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
        logger.error(f"Meta prompt generation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/strict-style', methods=['POST'])
def generate_strict_style():
    try:
        data = request.json
        if not data or 'topic' not in data or 'style_text' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
            
        model_choice = data.get('model', 'gemini')
        generator = StrictStyleContentGenerator(model_choice)
        
        result = generator.generate_style_matched_content(
            data['topic'],
            data['style_text'],
            max_retries=data.get('max_retries', 3)
        )
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        logger.error(f"Strict style generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/optimize-engagement', methods=['POST'])
def optimize_engagement():
    try:
        data = request.json
        if not data or 'content' not in data or 'platform' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
            
        model_choice = data.get('model', 'gemini')
        optimizer = EngagementOptimizer(model_choice)
        
        optimized_content = optimizer.optimize_engagement(
            data['content'],
            data['platform']
        )
        
        return jsonify({
            'success': True,
            'optimized_content': optimized_content
        })
        
    except Exception as e:
        logger.error(f"Engagement optimization error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-engagement', methods=['POST'])
def analyze_engagement():
    try:
        data = request.json
        if not data or 'content' not in data or 'platform' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
            
        model_choice = data.get('model', 'gemini')
        generator = ContentGenerator(model_choice)
        
        prompt = f"""
        Analyze the engagement potential of this content for {data['platform']}:
        
        Content: {data['content']}
        
        Evaluate:
        1. Hook strength
        2. Call-to-action effectiveness
        3. AIDA framework implementation
        4. Platform-specific optimization
        5. Potential engagement metrics
        
        Provide specific recommendations for improvement.
        """
        
        analysis = generator.generate_content(prompt)
        
        return jsonify({
            'success': True,
            'engagement_analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Engagement analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File is too large'}), 413

# Configuration for different environments
class Config:
    DEBUG = False
    TESTING = False
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    # Add production-specific config
    pass

class TestingConfig(Config):
    TESTING = True

# Configure the app based on environment
def configure_app(app: Flask) -> None:
    env = os.getenv('FLASK_ENV', 'development')
    if env == 'production':
        app.config.from_object(ProductionConfig)
    elif env == 'testing':
        app.config.from_object(TestingConfig)
    else:
        app.config.from_object(DevelopmentConfig)

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if __name__ == '__main__':
    # Configure the app
    configure_app(app)
    setup_logging()
    # Get port from environment variable or use default
    port = int(os.getenv('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port)