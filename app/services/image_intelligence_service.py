"""
Image Intelligence Service
Provides advanced image processing with OCR (Google Vision) and AI captions (GPT-5 Vision)
"""
from __future__ import annotations

import base64
import io
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import numpy as np

from google.cloud import vision
from google.oauth2 import service_account
from google.api_core import client_options
import openai

logger = logging.getLogger(__name__)


@dataclass
class ImageAnalysis:
    """Results from image analysis"""
    image_id: str
    page_number: int
    bbox: Optional[Dict[str, float]]  # Bounding box coordinates
    
    # OCR Results
    ocr_text: Optional[str]
    ocr_confidence: float
    detected_languages: List[str]
    
    # AI Caption
    ai_caption: Optional[str]
    caption_confidence: float
    
    # Visual Features
    dominant_colors: List[str]
    labels: List[Dict[str, float]]  # label -> confidence
    objects: List[Dict[str, Any]]  # Detected objects with bounding boxes
    
    # Combined searchable text
    searchable_text: str
    
    # Metadata
    width: int
    height: int
    format: str
    size_bytes: int


class ImageIntelligenceService:
    """Advanced image processing with OCR and AI captions"""
    
    def __init__(self):
        # Initialize Google Vision
        self.vision_client = None
        
        # Try API key first (simpler authentication)
        google_api_key = os.getenv('GOOGLE_VISION_API_KEY')
        if google_api_key:
            try:
                # Set API key for Google Vision
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''  # Clear any service account
                from google.api_core import client_options
                
                # Create client with API key
                options = client_options.ClientOptions(
                    api_key=google_api_key
                )
                self.vision_client = vision.ImageAnnotatorClient(client_options=options)
                logger.info("Google Vision API initialized successfully with API key")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Vision with API key: {e}")
        
        # Fallback to service account if API key not available
        if not self.vision_client:
            google_creds_path = os.getenv('GOOGLE_VISION_CREDENTIALS_PATH')
            if google_creds_path and os.path.exists(google_creds_path):
                try:
                    credentials = service_account.Credentials.from_service_account_file(
                        google_creds_path
                    )
                    self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                    logger.info("Google Vision API initialized successfully with service account")
                except Exception as e:
                    logger.warning(f"Failed to initialize Google Vision with service account: {e}")
            else:
                logger.warning("Google Vision not configured - set GOOGLE_VISION_API_KEY or GOOGLE_VISION_CREDENTIALS_PATH")
        
        # Initialize OpenAI for GPT-5 Vision
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.use_gpt5 = True  # Try GPT-5 first, fallback to GPT-4 if needed
        
    async def analyze_image(
        self,
        image_bytes: bytes,
        image_id: str,
        page_number: int = 1,
        context: Optional[str] = None,
        bbox: Optional[Dict[str, float]] = None
    ) -> ImageAnalysis:
        """
        Comprehensive image analysis with OCR and AI captions
        
        Args:
            image_bytes: Raw image data
            image_id: Unique identifier for the image
            page_number: Page number in document
            context: Document context for better captions
            bbox: Bounding box coordinates in document
            
        Returns:
            ImageAnalysis with all extracted information
        """
        # Load image with PIL for metadata
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        format_type = image.format or 'UNKNOWN'
        
        # Perform OCR with Google Vision
        ocr_result = await self._perform_ocr(image_bytes) if self.vision_client else {
            'text': '',
            'confidence': 0.0,
            'languages': []
        }
        
        # Generate AI caption with GPT-5 Vision
        ai_caption_result = await self._generate_ai_caption(
            image_bytes, 
            context=context,
            ocr_text=ocr_result['text']
        )
        
        # Extract visual features with Google Vision
        visual_features = await self._extract_visual_features(image_bytes) if self.vision_client else {
            'labels': [],
            'objects': [],
            'colors': []
        }
        
        # Combine all text for searchability
        searchable_parts = []
        if ocr_result['text']:
            searchable_parts.append(f"OCR: {ocr_result['text']}")
        if ai_caption_result['caption']:
            searchable_parts.append(f"Caption: {ai_caption_result['caption']}")
        if visual_features['labels']:
            label_text = ', '.join([l['description'] for l in visual_features['labels'][:5]])
            searchable_parts.append(f"Labels: {label_text}")
        
        searchable_text = ' | '.join(searchable_parts)
        
        return ImageAnalysis(
            image_id=image_id,
            page_number=page_number,
            bbox=bbox,
            ocr_text=ocr_result['text'],
            ocr_confidence=ocr_result['confidence'],
            detected_languages=ocr_result['languages'],
            ai_caption=ai_caption_result['caption'],
            caption_confidence=ai_caption_result['confidence'],
            dominant_colors=visual_features['colors'],
            labels=visual_features['labels'],
            objects=visual_features['objects'],
            searchable_text=searchable_text,
            width=width,
            height=height,
            format=format_type,
            size_bytes=len(image_bytes)
        )
    
    async def _perform_ocr(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Perform OCR using Google Vision API
        
        Returns:
            Dict with text, confidence, and detected languages
        """
        if not self.vision_client:
            return {'text': '', 'confidence': 0.0, 'languages': []}
        
        try:
            image = vision.Image(content=image_bytes)
            
            # Perform text detection
            response = self.vision_client.text_detection(image=image)
            texts = response.text_annotations
            
            if texts:
                # First annotation contains the entire detected text
                full_text = texts[0].description
                
                # Calculate average confidence from word-level annotations
                confidences = []
                for text in texts[1:]:  # Skip first (full text)
                    if hasattr(text, 'confidence'):
                        confidences.append(text.confidence)
                
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.9
                
                # Detect languages
                languages = []
                if response.full_text_annotation:
                    for page in response.full_text_annotation.pages:
                        if page.property and page.property.detected_languages:
                            for lang in page.property.detected_languages:
                                if lang.language_code not in languages:
                                    languages.append(lang.language_code)
                
                return {
                    'text': full_text.strip(),
                    'confidence': avg_confidence,
                    'languages': languages or ['en']
                }
            
            return {'text': '', 'confidence': 0.0, 'languages': []}
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return {'text': '', 'confidence': 0.0, 'languages': []}
    
    async def _generate_ai_caption(
        self, 
        image_bytes: bytes,
        context: Optional[str] = None,
        ocr_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate intelligent caption using GPT-5 Vision (August 2025 model)
        
        Returns:
            Dict with caption and confidence
        """
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Build context-aware prompt
            prompt_parts = [
                "Analyze this image and provide a detailed, searchable description."
            ]
            
            if context:
                prompt_parts.append(f"Document context: {context[:500]}")
            
            if ocr_text:
                prompt_parts.append(f"OCR detected text: {ocr_text[:200]}")
            
            prompt_parts.extend([
                "Focus on:",
                "1. What the image shows (diagrams, charts, photos, illustrations)",
                "2. Key information conveyed",
                "3. Relevant technical details",
                "4. How it relates to the document context",
                "Keep the description concise but comprehensive for search purposes."
            ])
            
            prompt = "\n".join(prompt_parts)
            
            # Use GPT-5 Vision (August 2025 - multimodal model)
            # GPT-5 DOES support vision when called with reasoning_effort parameter
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-5",  # Current best multimodal model
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high"  # High detail analysis
                                    }
                                }
                            ]
                        }
                    ],
                    max_completion_tokens=2000,  # Generous limit for detailed descriptions
                    temperature=1.0,
                    reasoning_effort="minimal"  # Required for GPT-5 to return content
                )
                model_used = "gpt-5"
            except Exception as e:
                # Try GPT-5-mini as lighter alternative
                logger.info(f"GPT-5 not available, trying GPT-5-mini: {e}")
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-5-mini",  # Faster GPT-5 variant
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                            "detail": "high"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_completion_tokens=2000,  # Generous limit for detailed descriptions
                        temperature=1.0,
                        reasoning_effort="minimal"  # Required for GPT-5 models
                    )
                    model_used = "gpt-5-mini"
                except Exception as e2:
                    # Try GPT-5-nano as ultra-light alternative
                    logger.info(f"GPT-5-mini not available, trying GPT-5-nano: {e2}")
                    try:
                        response = self.openai_client.chat.completions.create(
                            model="gpt-5-nano",  # Ultra-light GPT-5
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{base64_image}",
                                                "detail": "high"
                                            }
                                        }
                                    ]
                                }
                            ],
                            max_completion_tokens=300,
                            temperature=1.0,
                            reasoning_effort="minimal"  # Required for GPT-5 models
                        )
                        model_used = "gpt-5-nano"
                    except Exception as e3:
                        # Only as absolute last resort, fall back to GPT-4o (last year's model)
                        logger.warning(f"All GPT-5 models unavailable, using last year's GPT-4o as fallback: {e3}")
                        response = self.openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{base64_image}",
                                                "detail": "high"
                                            }
                                        }
                                    ]
                                }
                            ],
                            max_tokens=2000  # Generous limit for GPT-4o fallback
                        )
                        model_used = "gpt-4o-fallback"
            
            caption = response.choices[0].message.content.strip()
            
            # Confidence scores based on model quality
            confidence_map = {
                "gpt-5": 0.98,           # Best current multimodal model (Aug 2025)
                "gpt-5-mini": 0.95,      # Lighter but still GPT-5
                "gpt-5-nano": 0.93,      # Ultra-light GPT-5
                "gpt-4o-fallback": 0.85  # Last year's model, still good as fallback
            }
            confidence = confidence_map.get(model_used, 0.7)
            
            logger.info(f"Generated caption using {model_used}: {caption[:100]}...")
            
            return {
                'caption': caption,
                'confidence': confidence,
                'model': model_used
            }
            
        except Exception as e:
            logger.error(f"AI caption generation failed: {e}")
            return {
                'caption': '',
                'confidence': 0.0,
                'model': 'none'
            }
    
    async def _extract_visual_features(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract visual features using Google Vision API
        
        Returns:
            Dict with labels, objects, and colors
        """
        if not self.vision_client:
            return {'labels': [], 'objects': [], 'colors': []}
        
        try:
            image = vision.Image(content=image_bytes)
            
            # Detect labels (general image content)
            label_response = self.vision_client.label_detection(image=image, max_results=10)
            labels = [
                {
                    'description': label.description,
                    'score': label.score,
                    'topicality': label.topicality
                }
                for label in label_response.label_annotations
            ]
            
            # Detect objects with bounding boxes
            object_response = self.vision_client.object_localization(image=image)
            objects = [
                {
                    'name': obj.name,
                    'score': obj.score,
                    'bbox': {
                        'x_min': vertex.normalized_vertices[0].x,
                        'y_min': vertex.normalized_vertices[0].y,
                        'x_max': vertex.normalized_vertices[2].x,
                        'y_max': vertex.normalized_vertices[2].y
                    } if obj.bounding_poly else None
                }
                for obj in object_response.localized_object_annotations
            ]
            
            # Detect dominant colors
            props_response = self.vision_client.image_properties(image=image)
            colors = []
            if props_response.image_properties_annotation.dominant_colors:
                for color in props_response.image_properties_annotation.dominant_colors.colors[:5]:
                    rgb = color.color
                    hex_color = '#{:02x}{:02x}{:02x}'.format(
                        int(rgb.red or 0),
                        int(rgb.green or 0),
                        int(rgb.blue or 0)
                    )
                    colors.append(hex_color)
            
            return {
                'labels': labels,
                'objects': objects,
                'colors': colors
            }
            
        except Exception as e:
            logger.error(f"Visual feature extraction failed: {e}")
            return {'labels': [], 'objects': [], 'colors': []}
    
    async def batch_analyze_images(
        self,
        images: List[Tuple[bytes, str, int]],
        context: Optional[str] = None
    ) -> List[ImageAnalysis]:
        """
        Analyze multiple images in batch
        
        Args:
            images: List of (image_bytes, image_id, page_number) tuples
            context: Shared document context
            
        Returns:
            List of ImageAnalysis results
        """
        results = []
        
        for image_bytes, image_id, page_num in images:
            try:
                analysis = await self.analyze_image(
                    image_bytes=image_bytes,
                    image_id=image_id,
                    page_number=page_num,
                    context=context
                )
                results.append(analysis)
                
            except Exception as e:
                logger.error(f"Failed to analyze image {image_id}: {e}")
                # Create minimal analysis on failure
                results.append(ImageAnalysis(
                    image_id=image_id,
                    page_number=page_num,
                    bbox=None,
                    ocr_text='',
                    ocr_confidence=0.0,
                    detected_languages=[],
                    ai_caption='',
                    caption_confidence=0.0,
                    dominant_colors=[],
                    labels=[],
                    objects=[],
                    searchable_text='',
                    width=0,
                    height=0,
                    format='UNKNOWN',
                    size_bytes=len(image_bytes)
                ))
        
        return results
    
    def create_image_embedding_payload(self, analysis: ImageAnalysis) -> Dict[str, Any]:
        """
        Create payload for Qdrant storage with all image metadata
        
        Args:
            analysis: ImageAnalysis result
            
        Returns:
            Dict with all metadata for vector storage
        """
        return {
            'image_id': analysis.image_id,
            'page_number': analysis.page_number,
            'ocr_text': analysis.ocr_text or '',
            'ai_caption': analysis.ai_caption or '',
            'searchable_text': analysis.searchable_text,
            'labels': [l['description'] for l in analysis.labels[:5]],
            'objects': [o['name'] for o in analysis.objects[:5]],
            'colors': analysis.dominant_colors[:3],
            'width': analysis.width,
            'height': analysis.height,
            'format': analysis.format,
            'size_bytes': analysis.size_bytes,
            'ocr_confidence': analysis.ocr_confidence,
            'caption_confidence': analysis.caption_confidence,
            'languages': analysis.detected_languages,
            'has_text': bool(analysis.ocr_text),
            'has_caption': bool(analysis.ai_caption)
        }


# Singleton instance
_image_intelligence_service = None

def get_image_intelligence_service() -> ImageIntelligenceService:
    """Get or create singleton instance"""
    global _image_intelligence_service
    if _image_intelligence_service is None:
        _image_intelligence_service = ImageIntelligenceService()
    return _image_intelligence_service