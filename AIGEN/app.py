from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
import json
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler
import time
import io
import requests
import PyPDF2
import docx
from typing import Optional, Dict, Any, List, Tuple
import uuid
import re

# Load environment variables before importing API clients
load_dotenv()

# Now import and configure API clients
import google.generativeai as genai
from openai import OpenAI  # Updated OpenAI client import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure API keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')

# Validate required API keys for media
if not PEXELS_API_KEY:
    logger.warning("PEXELS_API_KEY not found in environment variables - media features will be unavailable")

# Initialize API clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    
# Initialize OpenAI client with newer client library
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# Replace the current setup_logging function with this
def setup_logging():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, 'app.log')
    
    # Create a more detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)
    file_handler.setFormatter(formatter)
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure app logger
    app.logger.setLevel(logging.INFO)
    
    # Remove default handlers to avoid duplication
    if app.logger.handlers:
        app.logger.handlers.clear()
    
    app.logger.addHandler(file_handler)
    app.logger.info('Application startup')
    
    # Log important configurations
    app.logger.info(f"Environment: {os.getenv('FLASK_ENV', 'development')}")
    app.logger.info(f"Gemini API Key configured: {bool(GEMINI_API_KEY)}")
    app.logger.info(f"OpenAI API Key configured: {bool(OPENAI_API_KEY)}")
    app.logger.info(f"Pexels API Key configured: {bool(PEXELS_API_KEY)}")

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ContentGenerator:
    def __init__(self, model_choice: str = 'gemini'):
        self.model_choice = model_choice.lower()
        if self.model_choice == 'gemini' and not GEMINI_API_KEY:
            logger.warning("Gemini API key not configured, will try OpenAI if available")
            if OPENAI_API_KEY:
                self.model_choice = 'openai'
            else:
                raise ValueError("No AI model API keys configured")
                
        if self.model_choice == 'openai' and not OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured, will try Gemini if available")
            if GEMINI_API_KEY:
                self.model_choice = 'gemini'
            else:
                raise ValueError("No AI model API keys configured")
            
    def list_available_models(self):
        """List available Gemini models"""
        try:
            if not GEMINI_API_KEY:
                return []
                
            available_models = genai.list_models()
            return [model.name for model in available_models]
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []

    # Replace the generate_content method in ContentGenerator class
    def generate_content(self, prompt: str) -> str:
        """
        Generates content using the specified AI model with enhanced error handling and retries.
        
        Args:
            prompt: The prompt to send to the AI model
            
        Returns:
            str: Generated content
            
        Raises:
            Exception: If content generation fails after all retries
        """
        retry_count = 0
        last_error = None
        
        while retry_count < MAX_RETRIES:
            try:
                logger.info(f"Content generation attempt {retry_count+1}/{MAX_RETRIES} with {self.model_choice}")
                logger.debug(f"Prompt (truncated): {prompt[:200]}...")
                
                start_time = time.time()
                
                # If Gemini is the first choice but OpenAI is available, we'll try Gemini first then fall back
                if self.model_choice == 'gemini':
                    try:
                        if not GEMINI_API_KEY:
                            raise ValueError("Gemini API key not configured")
                            
                        # Get available models to find the correct Gemini model name
                        available_models = self.list_available_models()
                        logger.info(f"Available Gemini models: {available_models}")
                        
                        # Try to find the best model to use
                        model_name = None
                        
                        # Look for the latest Gemini models in preferred order
                        for preferred_model in ['gemini-1.5-pro', 'gemini-1.0-pro', 'gemini-pro']:
                            for available_model in available_models:
                                if preferred_model in available_model:
                                    model_name = available_model
                                    break
                            if model_name:
                                break
                        
                        # If no specific model was found, try with the original name
                        if not model_name:
                            model_name = 'gemini-pro'
                            
                        logger.info(f"Using Gemini model: {model_name}")
                        model = genai.GenerativeModel(model_name)
                        
                        # Generate content (simplified safety settings)
                        generation_config = {
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "top_k": 40,
                            "max_output_tokens": 1000,
                        }
                        
                        response = model.generate_content(
                            prompt,
                            generation_config=generation_config
                        )
                        
                        if hasattr(response, 'text'):
                            content = response.text
                        # Fallback for different response structure
                        elif hasattr(response, 'parts'):
                            content = ''.join(part.text for part in response.parts)
                        else:
                            raise ValueError(f"Unexpected Gemini response format: {response}")
                        
                        # Log success
                        elapsed_time = time.time() - start_time
                        logger.info(f"Gemini content generation successful in {elapsed_time:.2f}s")
                        logger.debug(f"Generated content (truncated): {content[:100]}...")
                        return content
                        
                    except Exception as gemini_error:
                        logger.warning(f"Gemini error: {str(gemini_error)}")
                        # Fall back to OpenAI if available
                        if OPENAI_API_KEY:
                            logger.info("Falling back to OpenAI")
                            self.model_choice = 'openai'
                        else:
                            raise gemini_error
                    
                if self.model_choice == 'openai':
                    if not OPENAI_API_KEY or not openai_client:
                        raise ValueError("OpenAI API key not configured")
                        
                    try:
                        # Using new OpenAI client
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",  # Updated to gpt-4o, fallback to gpt-4 or gpt-3.5-turbo if needed
                            messages=[
                                {"role": "system", "content": "You are a professional content creator."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=1000
                        )
                        
                        content = response.choices[0].message.content
                        
                        # Log success
                        elapsed_time = time.time() - start_time
                        logger.info(f"OpenAI content generation successful in {elapsed_time:.2f}s")
                        logger.debug(f"Generated content (truncated): {content[:100]}...")
                        return content
                        
                    except Exception as openai_error:
                        logger.error(f"OpenAI error: {str(openai_error)}")
                        raise openai_error
                    
                else:
                    raise ValueError(f"Unsupported model choice: {self.model_choice}")
                    
            except Exception as e:
                retry_count += 1
                last_error = e
                logger.error(f"Content generation attempt {retry_count} failed: {str(e)}")
                
                if retry_count < MAX_RETRIES:
                    # Exponential backoff
                    wait_time = 2 ** retry_count
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {MAX_RETRIES} content generation attempts failed")
                    raise Exception(f"Failed to generate content after {MAX_RETRIES} attempts: {str(last_error)}")

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
                'tone': 'authoritative yet accessible professional tone',
                'length': 'between 1200-1600 characters (approx. 200-250 words)',
                'format': 'include 2-3 paragraphs with strategic whitespace and 1-2 bullet points for key takeaways',
                'elements': 'industry-specific insight, 1-2 data points, thought leadership positioning, focused call-to-action'
            },
            'twitter': {
                'tone': 'direct, conversational with personality',
                'length': 'maximum 280 characters (approx. 55-60 words)',
                'format': 'single impactful statement or question with 2-3 strategic hashtags',
                'elements': 'attention hook in first 3-4 words, one key point, clear CTA or question'
            },
            'instagram': {
                'tone': 'authentic, story-driven with visual language',
                'length': 'optimal 800-1000 characters (approx. 125-150 words)',
                'format': 'short paragraphs (1-2 sentences each) with line breaks and strategic emojis',
                'elements': 'sensory details, personal connection element, explicit reference to visual, 5-7 hashtags'
            },
            'facebook': {
                'tone': 'conversational, community-focused with personal elements',
                'length': 'optimal 1000-1500 characters (approx. 150-225 words)',
                'format': 'paragraph structure with question element and moderate emoji use',
                'elements': 'relatable story component, emotional hook, engagement question, clear next step'
            }
        }

        guidelines = platform_guidelines.get(platform.lower(), {
            'tone': 'balanced professional tone with clarity',
            'length': 'platform-appropriate length with focus on quality over quantity',
            'format': 'well-structured with appropriate whitespace and formatting',
            'elements': 'key messaging, unique value proposition, and contextual engagement elements'
        })

        meta_prompt = f"""
        PLATFORM: {platform.upper()}
        CONTENT TYPE: {content_type.upper()}
        TOPIC: {idea}

        TASK: Create a multi-step process to generate high-performing {platform} content about this topic.

        STEP 1: PLATFORM-SPECIFIC STRATEGY
        - Primary audience psychographics for this topic on {platform}
        - Content tone: {guidelines['tone']}
        - Optimal length: {guidelines['length']}
        - Structural format: {guidelines['format']}
        - Required platform-specific elements: {guidelines['elements']}
        
        STEP 2: CONTENT OPTIMIZATION FRAMEWORK
        - Hook strategy tailored to topic and platform
        - Value proposition articulation
        - Specific engagement triggers for this topic
        - Call-to-action optimization
        - Strategic use of platform-specific features 
        
        STEP 3: PERFORMANCE ENHANCEMENT
        - SEO/discoverability considerations for {platform}
        - Emotional response targeting
        - Specific metrics this content should optimize for
        - A/B testing variations to consider

        Combine these elements into a comprehensive, detailed prompt that would guide an AI to create the most effective possible {platform} content on this topic. The prompt should be specific, actionable, and focused on generating exactly ONE high-performance post.
        """

        try:
            logger.info(f"Generating meta-prompt for {platform} about: {idea}")
            optimized_prompt = self.generator.generate_content(meta_prompt)
            if not optimized_prompt:
                raise ValueError("Failed to generate optimized prompt")
                
            logger.info("Meta-prompt generation successful, now generating content")
            logger.debug(f"Meta-prompt (truncated): {optimized_prompt[:200]}...")

            # Generate final content using the optimized prompt
            final_content = self.generator.generate_content(optimized_prompt)
            if not final_content:
                raise ValueError("Failed to generate final content")
                
            logger.info("Final content generation successful")
            logger.debug(f"Generated content (truncated): {final_content[:200]}...")

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
        """
        Validates that the generated content meets platform-specific guidelines.
        
        Args:
            content (str): The content to validate
            platform (str): The platform for which content was generated
            
        Returns:
            dict: Validation result including success status and any notes
        """
        try:
            logger.info(f"Validating content for {platform}")
            
            # Get validation criteria based on platform
            validation_criteria = {
                'linkedin': {
                    'max_length': 3000,
                    'required_elements': ['professional tone', 'call-to-action'],
                },
                'twitter': {
                    'max_length': 280,
                    'required_elements': ['concise message', 'hashtags'],
                },
                'instagram': {
                    'max_length': 2200,
                    'required_elements': ['visual language', 'hashtags'],
                },
                'facebook': {
                    'max_length': 5000,
                    'required_elements': ['engagement question', 'call-to-action'],
                }
            }
            
            # Get platform-specific criteria or use default
            criteria = validation_criteria.get(platform.lower(), {
                'max_length': 3000,
                'required_elements': ['clear message'],
            })
            
            # Basic validation checks
            content_length = len(content)
            if content_length > criteria['max_length']:
                logger.warning(f"Content exceeds {platform} length limit: {content_length} > {criteria['max_length']}")
                return {
                    'success': False,
                    'error': f"Content exceeds {platform} maximum length ({content_length} > {criteria['max_length']})",
                    'recommendations': "Consider shortening the content to fit platform guidelines."
                }
                
            # More detailed validation could be implemented here
            # For example, checking for required elements or using AI to evaluate content quality
            
            # For now, basic validation passes
            logger.info(f"Content validation successful for {platform}")
            return {
                'success': True,
                'notes': f"Content meets basic {platform} requirements",
                'content_length': content_length,
                'max_length': criteria['max_length']
            }
            
        except Exception as e:
            logger.error(f"Content validation error: {str(e)}")
            return {
                'success': False,
                'error': f"Validation error: {str(e)}"
            }
# Replace the QAAgent class
class QAAgent:
    def __init__(self, model_choice: str = 'gemini'):
        self.generator = ContentGenerator(model_choice)
        self.max_retries = 3
        
    def validate_content(self, 
                        generated_content: str, 
                        original_style: str, 
                        topic: str) -> Tuple[bool, float, str]:
        """
        Improved content validation with more detailed scoring.
        
        Returns:
            Tuple of (is_valid, confidence_score, feedback)
        """
        # Construct a more precise validation prompt
        prompt = f"""
        Analyze these two pieces of content and determine style match accuracy:
        
        ORIGINAL STYLE EXAMPLE:
        ```
        {original_style}
        ```
        
        GENERATED CONTENT ON "{topic}":
        ```
        {generated_content}
        ```
        
        Score the following on a scale of 0-100:
        1. Tone and voice match: [score 0-100]
        2. Sentence structure similarity: [score 0-100]
        3. Vocabulary level/choice match: [score 0-100]
        4. Rhetorical devices/writing pattern match: [score 0-100]
        5. Overall style coherence: [score 0-100]
        
        TOTAL STYLE MATCH PERCENTAGE: [calculate the average of all scores]
        
        Provide specific examples from the text that illustrate where the style matches well and where it differs.
        Also provide clear, actionable feedback on how to improve the style matching.
        
        End your analysis with "FINAL_SCORE: [number]" where [number] is the total style match percentage.
        """
        
        try:
            logger.info("Starting content validation")
            analysis = self.generator.generate_content(prompt)
            
            # Extract the confidence score from the analysis
            confidence = 0.5  # Default middle value
            
            # Try to parse the final score
            try:
                if "FINAL_SCORE:" in analysis:
                    score_text = analysis.split("FINAL_SCORE:")[1].strip()
                    score_text = score_text.split()[0].rstrip('%.') # Handle different formats
                    confidence = float(score_text) / 100.0
                    logger.info(f"Extracted confidence score: {confidence:.2f}")
                else:
                    # Try to find any percentage in the last few lines
                    last_few_lines = analysis.split('\n')[-5:]
                    for line in last_few_lines:
                        if '%' in line:
                            score_match = re.search(r'(\d+)%', line)
                            if score_match:
                                confidence = float(score_match.group(1)) / 100.0
                                logger.info(f"Extracted fallback confidence score: {confidence:.2f}")
                                break
            except Exception as e:
                logger.warning(f"Error extracting confidence score: {str(e)}")
                logger.debug(f"Analysis text (truncated): {analysis[:200]}...")
                
            # Decide if the content is valid based on confidence threshold
            needs_revision = confidence < 0.85
            
            return not needs_revision, confidence, analysis
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False, 0.0, f"Validation failed: {str(e)}"

# Replace the StrictStyleContentGenerator class
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
        feedback_history = []
        
        # First analyze the style to understand it better
        style_analyzer = StyleAnalyzer(self.generator.model_choice)
        style_analysis = style_analyzer.analyze_style(style_text)
        logger.info(f"Style analysis complete. Proceeding with style matching.")
        
        while attempts < max_retries:
            try:
                logger.info(f"Style matching attempt {attempts+1}/{max_retries}")
                
                # Create an improved prompt that incorporates style analysis and previous feedback
                prompt = f"""
                Write content about: {topic}
                
                Match this writing style EXACTLY:
                ----
                {style_text}
                ----
                
                Style analysis:
                {style_analysis}
                
                {f"Previous feedback to incorporate: {feedback_history[-1]}" if feedback_history else ""}
                
                VERY IMPORTANT INSTRUCTIONS:
                1. Maintain exactly the same tone, voice, and perspective as the sample
                2. Copy the sentence structure patterns and paragraph length
                3. Use similar vocabulary level and terminology preferences
                4. Match rhetorical devices, idioms, and writing quirks
                5. Keep the same balance of description vs. dialogue or assertions
                6. Maintain similar use of punctuation, formatting, and emphasis
                7. Match the overall length and density of information
                
                DO NOT explain what you're going to write or discuss the style - simply write in the same style on the given topic.
                """
                
                content = self.generator.generate_content(prompt)
                
                # Log the generated content attempt
                logger.debug(f"Content attempt {attempts+1} (truncated): {content[:100]}...")
                
                # Validate how well it matches
                is_valid, confidence, feedback = self.qa_agent.validate_content(
                    content, style_text, topic
                )
                
                logger.info(f"Attempt {attempts+1} validation: confidence={confidence:.2f}, is_valid={is_valid}")
                
                feedback_history.append(feedback)
                
                if is_valid:
                    return {
                        "content": content,
                        "confidence": confidence,
                        "attempts": attempts + 1,
                        "style_analysis": style_analysis
                    }
                
                if confidence > best_confidence:
                    best_content = content
                    best_confidence = confidence
                    
                attempts += 1
                
            except Exception as e:
                logger.error(f"Error in style matching attempt {attempts+1}: {str(e)}")
                attempts += 1
        
        # If we've exhausted retries, return best attempt
        logger.warning(f"Style matching exhausted all {max_retries} attempts. Using best match with confidence {best_confidence:.2f}")
        return {
            "content": best_content,
            "confidence": best_confidence,
            "attempts": attempts,
            "warning": "Maximum retries reached. Using best attempt.",
            "style_analysis": style_analysis
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
            if not self.pexels_api_key:
                logger.warning("Pexels API key not configured, cannot fetch media")
                return None
                
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

    # Replace the platform_guidelines in ContentManager.generate_platform_content
    def generate_platform_content(self, platform: str, topic: str, preferences: str = "") -> str:
        platform_guidelines = {
            'linkedin': (
                "Professional tone with industry-specific terminology. "
                "Structure: 1) Hook with industry insight, 2) Three paragraphs max with value proposition, "
                "3) Include 1-2 data points or statistics, 4) Close with thought leadership position, "
                "5) One focused call-to-action, 6) 2-3 relevant hashtags only. "
                "Keep posts under 1300 characters. Use line breaks between paragraphs. "
                "Avoid excessive emoji, casual language, or overly promotional tone."
            ),
            'twitter': (
                "Conversational, direct tone with concise sentences. "
                "Structure: 1) Attention-grabbing first line with clear value, "
                "2) One key point only, 3) Strong call-to-action or question, "
                "4) 2-3 relevant hashtags maximum. "
                "Strict 280 character limit. Use abbreviations strategically. "
                "Incorporate current terminology relevant to topic. Avoid thread formats unless specified."
            ),
            'instagram': (
                "Engaging, visual-first narrative with emotional connection. "
                "Structure: 1) Opening line that creates curiosity, 2) Story-driven middle with sensory details, "
                "3) Personal connection element, 4) Clear call-to-action, "
                "5) 5-7 relevant hashtags grouped at the end. "
                "Keep under 1200 characters with strategic emojis. "
                "Use line breaks between every 1-2 sentences for visual spacing."
            ),
            'facebook': (
                "Balanced, community-focused tone with personal elements. "
                "Structure: 1) Relatable opening that creates connection, "
                "2) Two paragraphs maximum with story element, 3) Question to drive engagement, "
                "4) Clear next step or call-to-action. "
                "Keep under 1500 characters. Use moderate emojis. "
                "Incorporate question or poll element when possible."
            )
        }

        # Get the default guidelines or use a generic one for unlisted platforms
        platform_guide = platform_guidelines.get(platform.lower(), 
            "Professional yet conversational tone. Clear structure with introduction, "
            "main points, and conclusion. Include appropriate call-to-action. "
            "Use formatting suitable for digital reading."
        )
        
        # Create a more focused prompt
        prompt = f"""
        Create a highly optimized {platform} post about: {topic}
        
        DETAILED PLATFORM GUIDELINES:
        {platform_guide}
        
        USER PREFERENCES:
        {preferences}
        
        CONTENT CREATION REQUIREMENTS:
        1. Strictly adhere to the platform format and character limits
        2. Focus on quality and depth rather than excessive length
        3. Create only ONE post (not multiple options)
        4. Include appropriate engagement elements for {platform}
        5. Ensure the content is ready to publish without additional editing
        6. Create content that would perform in the top 10% of {platform} posts
        
        THE POST MUST BE IN THE EXACT FORMAT READY TO COPY-PASTE TO {platform.upper()}.
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
        # Ensure the upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
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
        request_id = str(uuid.uuid4())[:8]  # Generate a unique ID for this request
        logger.info(f"[{request_id}] Received request to /generate endpoint")
        
        # Log request headers and data for debugging
        headers = dict(request.headers)
        sensitive_headers = ['Authorization', 'Cookie']
        for h in sensitive_headers:
            if h in headers:
                headers[h] = '[REDACTED]'
        logger.debug(f"[{request_id}] Request headers: {headers}")
        
        try:
            data = request.json
            logger.debug(f"[{request_id}] Request data: {data}")
        except Exception as e:
            logger.error(f"[{request_id}] Error parsing request JSON: {str(e)}")
            return jsonify({'error': 'Invalid JSON in request body'}), 400
            
        if not data:
            logger.warning(f"[{request_id}] No data provided in request")
            return jsonify({'error': 'No data provided'}), 400

        required_fields = ['mode', 'platform', 'topic', 'model']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.warning(f"[{request_id}] Missing required fields: {missing_fields}")
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

        model_choice = data.get('model', 'gemini')
        logger.info(f"[{request_id}] Using model choice: {model_choice}")
        
        try:
            manager = ContentManager(model_choice)
        except Exception as e:
            logger.error(f"[{request_id}] Failed to initialize ContentManager: {str(e)}")
            return jsonify({'error': f'Failed to initialize AI models: {str(e)}'}), 500

        # Record start time for performance tracking
        start_time = time.time()

        # Generate content based on mode
        try:
            if data['mode'] == 'strict_style' and data.get('style_text'):
                logger.info(f"[{request_id}] Generating content with strict style matching")
                strict_generator = StrictStyleContentGenerator(model_choice)
                result = strict_generator.generate_style_matched_content(
                    data['topic'],
                    data['style_text']
                )
                content = result['content']
                
                # Add diagnostic information
                generation_info = {
                    'confidence': result.get('confidence', 0),
                    'attempts': result.get('attempts', 0),
                    'warnings': result.get('warning', None)
                }
            else:
                logger.info(f"[{request_id}] Generating platform content for {data['platform']}")
                content = manager.generate_platform_content(
                    data['platform'],
                    data['topic'],
                    data.get('preferences', '')
                )
                generation_info = {'mode': 'platform_standard'}
                
        except Exception as e:
            logger.error(f"[{request_id}] Content generation failed: {str(e)}", exc_info=True)
            return jsonify({
                'error': f'Content generation failed: {str(e)}',
                'success': False
            }), 500

        generation_time = time.time() - start_time
        logger.info(f"[{request_id}] Content generated in {generation_time:.2f}s")
        
        # Optimize engagement if requested
        if data.get('optimize_engagement', False):
            try:
                logger.info(f"[{request_id}] Optimizing engagement")
                optimizer = EngagementOptimizer(model_choice)
                content = optimizer.optimize_engagement(content, data['platform'])
                logger.info(f"[{request_id}] Engagement optimization complete")
            except Exception as e:
                logger.error(f"[{request_id}] Engagement optimization failed: {str(e)}")
                # Continue with unoptimized content rather than failing completely
                generation_info['engagement_optimization_error'] = str(e)

        # Handle media if requested
        media_url = None
        if data.get('media_type') in ['image', 'video']:
            try:
                logger.info(f"[{request_id}] Fetching {data['media_type']} media")
                media_url = manager.multimedia.get_media(
                    data['topic'],
                    data['media_type']
                )
                logger.info(f"[{request_id}] Media fetch result: {'successful' if media_url else 'no media found'}")
            except Exception as e:
                logger.error(f"[{request_id}] Media fetch failed: {str(e)}")
                # Continue without media rather than failing completely
                generation_info['media_fetch_error'] = str(e)

        # Final response timing
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] Request completed in {total_time:.2f}s")

        return jsonify({
            'success': True,
            'content': content,
            'media_url': media_url,
            'generation_info': generation_info,
            'processing_time_seconds': round(total_time, 2)
        })

    except Exception as e:
        logger.error(f"Unexpected error in generate endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/meta-prompt', methods=['POST'])
def generate_meta_prompt():
    try:
        # Log incoming request
        logger.debug(f"Request headers: {dict(request.headers)}")
        # Get data and handle potential JSON parsing issues
        try:
            data = request.get_json(force=True)
        except Exception as e:
            logger.error(f"JSON parsing error: {str(e)}")
            # Try to log the raw data for debugging
            raw_data = request.get_data().decode('utf-8')
            logger.debug(f"Raw request data: {raw_data[:500]}...")  # Log first 500 chars
            return jsonify({'success': False, 'error': 'Invalid JSON data'}), 400
            
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
        
        logger.info(f"Generating meta-prompt for {platform} content about {topic}")
        
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