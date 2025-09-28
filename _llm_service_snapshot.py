"""LLM Service for multi-model support with OpenAI and Gemini"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import TimeoutError
import time

from openai import OpenAI, AsyncOpenAI
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    GEMINI = "gemini"
    
class LLMModel(Enum):
    """Available models from each provider"""
    # OpenAI models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_35_TURBO = "gpt-3.5-turbo"
    
    # GPT-5 models (Released August 2025, knowledge cutoff Sept 2024)
    # GPT-5 is a unified system with smart routing between models
    GPT_5 = "gpt-5"  # Standard model, balanced reasoning and speed
    GPT_5_THINKING = "gpt-5-thinking"  # Extended reasoning mode (aka GPT-5 Pro) 
    GPT_5_MINI = "gpt-5-mini"  # Faster, cheaper for everyday queries
    GPT_5_NANO = "gpt-5-nano"  # Ultra-lightweight for mobile/embedded
    
    # Gemini models
    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_25_FLASH = "gemini-2.5-flash"
    GEMINI_20_FLASH = "gemini-2.0-flash"
    GEMINI_20_FLASH_LITE = "gemini-2.0-flash-lite"
    GEMINI_15_PRO = "gemini-1.5-pro"  # Keep for backwards compatibility
    GEMINI_15_FLASH = "gemini-1.5-flash"  # Keep for backwards compatibility

@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    provider: LLMProvider
    model: str
    usage: Dict[str, int]
    latency_ms: int
    cost_estimate: float = 0.0
    
@dataclass
class Entity:
    """Entity extracted from text"""
    name: str
    type: str
    confidence: float
    context: str
    
@dataclass
class DocumentMetadata:
    """Metadata extracted from document"""
    title: Optional[str] = None
    author: Optional[str] = None
    department: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = None
    summary: Optional[str] = None
    key_topics: List[str] = None
    sentiment: Optional[str] = None
    confidence: float = 0.0

class LLMService:
    """Multi-model LLM service with fallback and comparison capabilities"""
    
    def __init__(self):
        """Initialize LLM clients"""
        # OpenAI setup
        if settings.openai_api_key:
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
            self.openai_async = AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            self.openai_client = None
            self.openai_async = None
            logger.warning("OpenAI API key not configured")
        
        # Gemini setup
        if settings.google_ai_api_key:
            genai.configure(api_key=settings.google_ai_api_key)
            self.gemini_client = genai.GenerativeModel('gemini-2.5-pro')
        else:
            self.gemini_client = None
            logger.warning("Gemini API key not configured")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((TimeoutError, ConnectionError))
    )
    async def call_llm(
        self,
        prompt: str,
        provider: LLMProvider = LLMProvider.OPENAI,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 30,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Call LLM with specified provider and model
        
        Args:
            prompt: User prompt
            provider: LLM provider to use
            model: Specific model to use
            temperature: Response randomness (0-1)
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds
            system_prompt: System instructions
            
        Returns:
            Standardized LLM response
        """
        start_time = time.time()
        
        try:
            if provider == LLMProvider.OPENAI:
                response = await self._call_openai(
                    prompt, model, temperature, max_tokens, timeout, system_prompt
                )
            elif provider == LLMProvider.GEMINI:
                response = await self._call_gemini(
                    prompt, model, temperature, max_tokens, timeout, system_prompt
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            response.latency_ms = latency_ms
            
            # Estimate cost
            response.cost_estimate = self._estimate_cost(
                response.provider, response.model, response.usage
            )
            
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"LLM call timed out after {timeout} seconds")
            raise TimeoutError(f"LLM call timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    async def _call_openai(
        self,
        prompt: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: int,
        system_prompt: Optional[str]
    ) -> LLMResponse:
        """Call OpenAI API"""
        if not self.openai_async:
            raise ValueError("OpenAI client not configured")
        
        model = model or LLMModel.GPT_4O_MINI.value
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Use asyncio timeout
        async with asyncio.timeout(timeout):
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            # Some models (e.g., gpt-5) do not support 'max_tokens' in Chat Completions
            if not str(model).startswith("gpt-5"):
                kwargs["max_tokens"] = max_tokens
            response = await self.openai_async.chat.completions.create(**kwargs)
        
        return LLMResponse(
            content=response.choices[0].message.content,
            provider=LLMProvider.OPENAI,
            model=model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            latency_ms=0  # Will be set by caller
        )
    
    async def _call_gemini(
        self,
        prompt: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: int,
        system_prompt: Optional[str]
    ) -> LLMResponse:
        """Call Gemini API"""
        if not self.gemini_client:
            raise ValueError("Gemini client not configured")
        
        # Combine system prompt with user prompt for Gemini
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Gemini requires minimum 1000 tokens to generate any output
        # If less than 1000, increase to minimum working value
        effective_max_tokens = max(max_tokens, 1000)
        if max_tokens < 1000:
            logger.info(f"Increasing Gemini max_tokens from {max_tokens} to {effective_max_tokens} (minimum required)")
        
        # Gemini uses different parameter names
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=effective_max_tokens
        )
        
        # Define less restrictive safety settings for business use cases
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
        }
        
        # Run in executor to make it async with timeout
        loop = asyncio.get_event_loop()
        
        async with asyncio.timeout(timeout):
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini_client.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
            )
        
        # Check if response was blocked or empty
        if not response.candidates or not response.candidates[0].content.parts:
            # Get the finish reason if available
            finish_reason = response.candidates[0].finish_reason if response.candidates else None
            
            # Handle numeric finish_reason (older API versions or direct proto values)
            if isinstance(finish_reason, int):
                reason_map = {
                    0: "FINISH_REASON_UNSPECIFIED",
                    1: "STOP",
                    2: "MAX_TOKENS",
                    3: "SAFETY",
                    4: "RECITATION",
                    5: "OTHER"
                }
                finish_reason_name = reason_map.get(finish_reason, f"UNKNOWN_{finish_reason}")
            elif finish_reason:
                finish_reason_name = finish_reason.name if hasattr(finish_reason, 'name') else str(finish_reason)
            else:
                finish_reason_name = "Unknown"
            
            if finish_reason_name == 'SAFETY':
                # Response was blocked due to safety
                safety_ratings = response.candidates[0].safety_ratings if response.candidates else []
                logger.warning(f"Gemini response blocked by safety filters. Ratings: {safety_ratings}")
                raise ValueError(f"Content blocked by safety filters. Finish reason: SAFETY")
            elif finish_reason_name == 'RECITATION':
                logger.warning("Gemini response blocked due to recitation concerns")
                raise ValueError("Content blocked due to potential copyright/recitation issues")
            elif finish_reason_name == 'MAX_TOKENS':
                logger.warning("Gemini hit max token limit with no content generated")
                # For MAX_TOKENS, there might be partial content
                try:
                    if response.candidates and response.candidates[0].content.parts:
                        response_text = response.text
                        logger.info(f"Partial content retrieved despite MAX_TOKENS: {len(response_text)} chars")
                        # Continue with partial content
                    else:
                        raise ValueError("Hit max token limit with no content generated")
                except:
                    raise ValueError("Hit max token limit with no content generated")
            else:
                # Handle other reasons for empty response
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    reason_str = f"PROMPT_BLOCKED ({response.prompt_feedback.block_reason.name})"
                else:
                    reason_str = finish_reason_name
                logger.error(f"Gemini returned empty response. Reason: {reason_str}")
                raise ValueError(f"Gemini returned empty response. Reason: {reason_str}")
        else:
            # Response has content, safe to access
            response_text = response.text
        
        # Extract token usage (Gemini doesn't provide exact counts)
        # Estimate based on text length
        prompt_tokens = len(full_prompt.split()) * 1.3
        completion_tokens = len(response_text.split()) * 1.3
        
        return LLMResponse(
            content=response_text,
            provider=LLMProvider.GEMINI,
            model=model or "gemini-2.5-pro",
            usage={
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens)
            },
            latency_ms=0  # Will be set by caller
        )
    
    def _estimate_cost(
        self,
        provider: LLMProvider,
        model: str,
        usage: Dict[str, int]
    ) -> float:
        """Estimate API call cost in USD"""
        # Pricing per 1K tokens (approximate as of 2024)
        pricing = {
            LLMProvider.OPENAI: {
                "gpt-4o": {"input": 0.005, "output": 0.015},
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
                "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
            },
            LLMProvider.GEMINI: {
                "gemini-2.5-pro": {"input": 0.0035, "output": 0.0105},
                "gemini-2.5-flash": {"input": 0.00035, "output": 0.00105},
                "gemini-2.0-flash": {"input": 0.00025, "output": 0.00075},
                "gemini-2.0-flash-lite": {"input": 0.0001, "output": 0.0003},
                "gemini-1.5-pro": {"input": 0.0035, "output": 0.0105},
                "gemini-1.5-flash": {"input": 0.00035, "output": 0.00105}
            }
        }
        
        if provider not in pricing or model not in pricing[provider]:
            return 0.0
        
        rates = pricing[provider][model]
        input_cost = (usage.get("prompt_tokens", 0) / 1000) * rates["input"]
        output_cost = (usage.get("completion_tokens", 0) / 1000) * rates["output"]
        
        return round(input_cost + output_cost, 6)
    
    async def extract_metadata(
        self,
        text: str,
        provider: LLMProvider = LLMProvider.OPENAI,
        model: Optional[str] = None
    ) -> DocumentMetadata:
        """
        Extract metadata from document text
        
        Args:
            text: Document text to analyze
            provider: LLM provider to use
            model: Specific model to use
            
        Returns:
            Extracted metadata
        """
        system_prompt = """You are a document analysis expert. Extract metadata from the given document text.
        Return a JSON object with the following fields:
        - title: Document title
        - author: Author name(s)
        - department: Relevant department or organization
        - category: Document category (e.g., report, proposal, manual)
        - tags: List of relevant tags (max 5)
        - summary: Brief summary (max 100 words)
        - key_topics: Main topics discussed (max 5)
        - sentiment: Overall sentiment (positive, neutral, negative)
        - confidence: Your confidence in the extraction (0-1)
        
        If a field cannot be determined, use null."""
        
        prompt = f"Extract metadata from this document:\n\n{text[:3000]}"  # Limit text length
        
        try:
            response = await self.call_llm(
                prompt=prompt,
                provider=provider,
                model=model,
                temperature=0.3,  # Lower temperature for structured extraction
                system_prompt=system_prompt,
                timeout=30
            )
            
            # Parse JSON response - handle both raw JSON and markdown-wrapped JSON
            content = response.content.strip()
            
            # Remove markdown code block if present (common with Gemini)
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
                if content.endswith("```"):
                    content = content[:-3]  # Remove closing ```
                content = content.strip()
            elif content.startswith("```"):
                content = content[3:]  # Remove opening ```
                if content.endswith("```"):
                    content = content[:-3]  # Remove closing ```
                content = content.strip()
            
            metadata_dict = json.loads(content)
            
            return DocumentMetadata(
                title=metadata_dict.get("title"),
                author=metadata_dict.get("author"),
                department=metadata_dict.get("department"),
                category=metadata_dict.get("category"),
                tags=metadata_dict.get("tags", []),
                summary=metadata_dict.get("summary"),
                key_topics=metadata_dict.get("key_topics", []),
                sentiment=metadata_dict.get("sentiment"),
                confidence=metadata_dict.get("confidence", 0.5)
            )
            
        except json.JSONDecodeError:
            logger.error("Failed to parse metadata JSON response")
            # Try to extract basic info from response text
            return DocumentMetadata(
                summary=response.content[:200] if response else None,
                confidence=0.3
            )
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return DocumentMetadata(confidence=0.0)
    
    async def extract_entities(
        self,
        text: str,
        provider: LLMProvider = LLMProvider.OPENAI,
        model: Optional[str] = None
    ) -> List[Entity]:
        """
        Extract entities from text
        
        Args:
            text: Text to analyze
            provider: LLM provider to use
            model: Specific model to use
            
        Returns:
            List of extracted entities
        """
        system_prompt = """You are building a knowledge graph from text. Extract entities that will become meaningful nodes in a graph where relationships between concepts are essential for understanding.

        Your goal is to identify concepts that:
        1. Users would search for to find this document
        2. Connect to other concepts through relationships
        3. Represent the core knowledge in the text

        ENTITY TYPES TO EXTRACT:
        - PERSON: Individuals with specific names
        - ORGANIZATION: Companies, institutions, agencies
        - LOCATION: Places, facilities, regions
        - DATE: Specific dates, time periods, deadlines
        - PRODUCT: Branded products, models, versions
        - COMPONENT: Physical or software parts that are part of larger systems
        - TECHNOLOGY: Platforms, frameworks, standards, protocols
        - CHEMICAL: Substances, compounds, materials
        - PROCEDURE: Methods, processes, techniques, workflows
        - SPECIFICATION: Standards, requirements, regulations, parameters
        - SYSTEM: Complex assemblies, integrated platforms, subsystems
        - MEASUREMENT: Quantities with units, dimensions, thresholds
        - PROBLEM: Issues, failures, errors, symptoms, defects
        - CONDITION: States of wear, degradation, or quality (corrosion, contamination, wear)
        - STATE: Operational modes or statuses (active, locked, failed, operational)
        - CONCEPT: Domain-specific abstract ideas
        - EVENT: Incidents, occurrences, milestones

        EXTRACTION GUIDELINES:
        - Focus on terms that can have relationships with other terms
        - Include both the thing and its state (e.g., "display" AND "flickering")
        - Extract symptoms AND their causes when identifiable
        - Include materials/tools AND what they're used for
        - Capture procedural steps that modify or assess things

        ADAPT TO DOCUMENT TYPE:
        - Technical troubleshooting: problems, symptoms, components, procedures, states
        - Specifications: measurements, requirements, components, systems
        - Procedures: steps, tools, materials, conditions to check
        - Analysis: findings, evidence, hypotheses, methods

        DO NOT EXTRACT:
        - Generic words without specific meaning
        - Pronouns or vague references
        - Common verbs unless they represent specific procedures

        Return a JSON array where each entity has:
        - name: The exact term as it appears
        - type: One of [PERSON, ORGANIZATION, LOCATION, DATE, PRODUCT, COMPONENT, TECHNOLOGY, CHEMICAL, PROCEDURE, SPECIFICATION, SYSTEM, MEASUREMENT, PROBLEM, CONDITION, STATE, CONCEPT, EVENT]
        - confidence: 0.0 to 1.0
        - context: Brief context where found (optional)
        
        Maximum 20 most important entities. Quality over quantity."""
        
        prompt = f"Extract meaningful entities from this text:\n\n{text[:3000]}"
        
        try:
            response = await self.call_llm(
                prompt=prompt,
                provider=provider,
                model=model,
                temperature=0.1,  # Very low temperature for consistent extraction
                system_prompt=system_prompt,
                timeout=30
            )
            
            # Parse JSON response - handle both raw JSON and markdown-wrapped JSON
            content = response.content.strip()
            
            # Remove markdown code block if present (common with Gemini)
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
                if content.endswith("```"):
                    content = content[:-3]  # Remove closing ```
                content = content.strip()
            elif content.startswith("```"):
                content = content[3:]  # Remove opening ```
                if content.endswith("```"):
                    content = content[:-3]  # Remove closing ```
                content = content.strip()
            
            entities_list = json.loads(content)
            
            # Common words to filter out (stop words and generic terms)
            stop_words = {
                'the', 'this', 'that', 'these', 'those', 'how', 'when', 'where', 'why', 'what',
                'who', 'which', 'can', 'could', 'would', 'should', 'may', 'might', 'must',
                'will', 'shall', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'and', 'or', 'but',
                'if', 'then', 'else', 'for', 'to', 'from', 'with', 'without', 'by', 'at',
                'in', 'on', 'up', 'down', 'out', 'off', 'over', 'under', 'between',
                'through', 'during', 'before', 'after', 'above', 'below', 'each', 'few',
                'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'also',
                'user', 'users', 'system', 'systems', 'data', 'information', 'document',
                'file', 'files', 'item', 'items', 'thing', 'things', 'way', 'ways'
            }
            
            filtered_entities = []
            for e in entities_list:
                name = e.get("name", "").strip()
                
                # Skip if no name or too short
                if not name or len(name) < 2:
                    continue
                
                # Skip if it's a stop word (case-insensitive check)
                if name.lower() in stop_words:
                    continue
                
                # Skip if confidence is too low
                confidence = e.get("confidence", 0.5)
                if confidence < 0.5:
                    continue
                
                # Skip single letters or numbers
                if len(name) == 1 and (name.isalpha() or name.isdigit()):
                    continue
                
                # Create entity with proper type mapping
                entity_type = e.get("type", "UNKNOWN").upper()
                
                filtered_entities.append(
                    Entity(
                        name=name,
                        type=entity_type,
                        confidence=confidence,
                        context=e.get("context", "")
                    )
                )
            
            # Sort by confidence and return top entities
            filtered_entities.sort(key=lambda x: x.confidence, reverse=True)
            return filtered_entities[:20]  # Limit to top 20
            
        except json.JSONDecodeError:
            logger.error("Failed to parse entities JSON response")
            return []
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def compare_outputs(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, LLMResponse]:
        """
        Get outputs from multiple models for comparison
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Response randomness
            max_tokens: Maximum response tokens
            
        Returns:
            Dictionary of provider -> response
        """
        results = {}
        
        # Try OpenAI
        if self.openai_async:
            try:
                results["openai"] = await self.call_llm(
                    prompt=prompt,
                    provider=LLMProvider.OPENAI,
                    model=LLMModel.GPT_4O_MINI.value,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    timeout=30
                )
            except Exception as e:
                logger.error(f"OpenAI comparison failed: {e}")
                results["openai_error"] = str(e)
        
        # Try Gemini
        if self.gemini_client:
            try:
                results["gemini"] = await self.call_llm(
                    prompt=prompt,
                    provider=LLMProvider.GEMINI,
                    model=LLMModel.GEMINI_25_PRO.value,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    timeout=30
                )
            except Exception as e:
                logger.error(f"Gemini comparison failed: {e}")
                results["gemini_error"] = str(e)
        
        return results
    
    async def extract_entities_with_comparison(
        self,
        text: str,
        use_gemini: bool = True
    ) -> Dict[str, Any]:
        """
        Extract entities using multiple models and compare results
        
        Args:
            text: Text to analyze
            use_gemini: Whether to include Gemini in comparison
            
        Returns:
            Dictionary with comparison results and consensus
        """
        results = {"openai": [], "gemini": [], "consensus": []}
        
        # Extract with OpenAI
        try:
            openai_entities = await self.extract_entities(
                text,
                provider=LLMProvider.OPENAI,
                model=LLMModel.GPT_4O_MINI.value
            )
            results["openai"] = [
                {
                    "name": e.name,
                    "type": e.type,
                    "confidence": e.confidence,
                    "context": e.context
                }
                for e in openai_entities
            ]
        except Exception as e:
            logger.error(f"OpenAI entity extraction failed: {e}")
            results["openai_error"] = str(e)
        
        # Extract with Gemini if requested
        if use_gemini and self.gemini_client:
            try:
                gemini_entities = await self.extract_entities(
                    text,
                    provider=LLMProvider.GEMINI,
                    model=LLMModel.GEMINI_25_PRO.value
                )
                results["gemini"] = [
                    {
                        "name": e.name,
                        "type": e.type,
                        "confidence": e.confidence,
                        "context": e.context
                    }
                    for e in gemini_entities
                ]
            except Exception as e:
                logger.error(f"Gemini entity extraction failed: {e}")
                results["gemini_error"] = str(e)
        
        # Create consensus list (entities found by at least one model)
        entity_map = {}
        
        # Add OpenAI entities
        for entity in results.get("openai", []):
            key = f"{entity['name'].lower()}_{entity['type']}"
            if key not in entity_map:
                entity_map[key] = entity.copy()
                entity_map[key]["sources"] = ["openai"]
            else:
                entity_map[key]["sources"].append("openai")
                # Update confidence if higher
                if entity["confidence"] > entity_map[key]["confidence"]:
                    entity_map[key]["confidence"] = entity["confidence"]
        
        # Add Gemini entities
        for entity in results.get("gemini", []):
            key = f"{entity['name'].lower()}_{entity['type']}"
            if key not in entity_map:
                entity_map[key] = entity.copy()
                entity_map[key]["sources"] = ["gemini"]
            else:
                entity_map[key]["sources"].append("gemini")
                # Update confidence if higher
                if entity["confidence"] > entity_map[key]["confidence"]:
                    entity_map[key]["confidence"] = entity["confidence"]
        
        # Sort by confidence and number of sources
        consensus_entities = list(entity_map.values())
        consensus_entities.sort(
            key=lambda x: (len(x.get("sources", [])), x["confidence"]),
            reverse=True
        )
        
        results["consensus"] = consensus_entities
        results["total_unique"] = len(consensus_entities)
        results["agreed_upon"] = len([e for e in consensus_entities if len(e.get("sources", [])) > 1])
        
        return results
    
    async def call_with_fallback(
        self,
        prompt: str,
        primary_provider: LLMProvider = LLMProvider.OPENAI,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Call LLM with automatic fallback to alternative provider
        
        Args:
            prompt: User prompt
            primary_provider: Preferred provider
            system_prompt: System instructions
            **kwargs: Additional arguments for call_llm
            
        Returns:
            LLM response from primary or fallback provider
        """
        # Try primary provider
        try:
            return await self.call_llm(
                prompt=prompt,
                provider=primary_provider,
                system_prompt=system_prompt,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Primary provider {primary_provider.value} failed: {e}")
            
            # Determine fallback provider
            fallback_provider = (
                LLMProvider.GEMINI 
                if primary_provider == LLMProvider.OPENAI 
                else LLMProvider.OPENAI
            )
            
            # Check if fallback is available
            if fallback_provider == LLMProvider.OPENAI and not self.openai_async:
                raise ValueError("No fallback provider available")
            if fallback_provider == LLMProvider.GEMINI and not self.gemini_client:
                raise ValueError("No fallback provider available")
            
            logger.info(f"Falling back to {fallback_provider.value}")
            
            # Try fallback provider
            return await self.call_llm(
                prompt=prompt,
                provider=fallback_provider,
                system_prompt=system_prompt,
                **kwargs
            )

