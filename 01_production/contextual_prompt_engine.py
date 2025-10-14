#!/usr/bin/env python3
"""
Contextual Prompt Generation Engine for RealityGuard
Revolutionary enhancement that makes AI replacement context-aware
Patent enhancement - adds intelligence to content generation
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib
from collections import defaultdict

# Try importing CLIP for scene understanding
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not installed - will use fallback scene analysis")

# Try importing transformers for advanced NLP
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not installed - will use basic prompts")


class SceneContext(Enum):
    """Different scene contexts that affect privacy generation."""
    OFFICE = "office"
    MEDICAL = "medical"
    EDUCATIONAL = "educational"
    HOME = "home"
    OUTDOOR = "outdoor"
    RETAIL = "retail"
    CONFERENCE = "conference"
    SOCIAL = "social"
    INDUSTRIAL = "industrial"
    UNKNOWN = "unknown"


class ObjectType(Enum):
    """Types of objects that need privacy protection."""
    PERSON = "person"
    FACE = "face"
    SCREEN = "screen"
    DOCUMENT = "document"
    LICENSE_PLATE = "license_plate"
    BADGE = "badge"
    WHITEBOARD = "whiteboard"
    PHONE = "phone"
    LAPTOP = "laptop"
    UNKNOWN = "object"


class EmotionalTone(Enum):
    """Emotional tone to preserve in generated content."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FORMAL = "formal"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    SERIOUS = "serious"
    NEUTRAL = "neutral"


@dataclass
class ContextualPrompt:
    """Structured prompt with all contextual information."""
    base_prompt: str
    negative_prompt: str
    style_tokens: List[str]
    preservation_hints: Dict[str, any]
    strength: float = 0.8
    guidance_scale: float = 7.5


class SceneAnalyzer:
    """Analyzes scene context using CLIP or fallback methods."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if CLIP_AVAILABLE:
            print("Loading CLIP model for scene understanding...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()

            # Precompute scene embeddings
            self._prepare_scene_embeddings()
        else:
            print("Using fallback scene analysis")
            self.clip_model = None

    def _prepare_scene_embeddings(self):
        """Precompute embeddings for different scene types."""
        self.scene_descriptions = {
            SceneContext.OFFICE: [
                "an office environment",
                "office workspace with computers",
                "business meeting room",
                "corporate office setting"
            ],
            SceneContext.MEDICAL: [
                "medical facility",
                "hospital room",
                "doctor's office",
                "healthcare setting"
            ],
            SceneContext.EDUCATIONAL: [
                "classroom",
                "school environment",
                "educational setting",
                "lecture hall"
            ],
            SceneContext.HOME: [
                "home interior",
                "living room",
                "residential space",
                "domestic environment"
            ],
            SceneContext.OUTDOOR: [
                "outdoor scene",
                "street view",
                "nature environment",
                "outside area"
            ],
            SceneContext.CONFERENCE: [
                "video conference",
                "zoom meeting",
                "virtual meeting",
                "video call"
            ]
        }

        if self.clip_model:
            self.scene_embeddings = {}
            with torch.no_grad():
                for context, descriptions in self.scene_descriptions.items():
                    texts = clip.tokenize(descriptions).to(self.device)
                    embeddings = self.clip_model.encode_text(texts)
                    embeddings /= embeddings.norm(dim=-1, keepdim=True)
                    self.scene_embeddings[context] = embeddings.mean(dim=0)

    def analyze_scene(self, frame: np.ndarray) -> Tuple[SceneContext, float]:
        """Analyze the scene context of a frame."""
        if self.clip_model:
            return self._analyze_with_clip(frame)
        else:
            return self._analyze_fallback(frame)

    def _analyze_with_clip(self, frame: np.ndarray) -> Tuple[SceneContext, float]:
        """Use CLIP to understand scene context."""
        # Preprocess frame for CLIP
        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

        # Get image embedding
        with torch.no_grad():
            image_embedding = self.clip_model.encode_image(image)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

        # Compare with scene embeddings
        best_context = SceneContext.UNKNOWN
        best_similarity = -1.0

        for context, scene_embedding in self.scene_embeddings.items():
            similarity = (image_embedding @ scene_embedding.T).item()
            if similarity > best_similarity:
                best_similarity = similarity
                best_context = context

        return best_context, best_similarity

    def _analyze_fallback(self, frame: np.ndarray) -> Tuple[SceneContext, float]:
        """Fallback scene analysis using traditional CV."""
        h, w = frame.shape[:2]

        # Analyze color distribution
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])

        # Normalize histograms
        hist_h = hist_h.flatten() / hist_h.sum()
        hist_s = hist_s.flatten() / hist_s.sum()
        hist_v = hist_v.flatten() / hist_v.sum()

        # Simple heuristics
        avg_saturation = np.average(np.arange(256), weights=hist_s)
        avg_value = np.average(np.arange(256), weights=hist_v)

        # Detect edges for structure
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (h * w)

        # Make educated guess based on features
        if edge_density > 0.15 and avg_saturation < 100:
            return SceneContext.OFFICE, 0.6
        elif avg_value > 180:
            return SceneContext.MEDICAL, 0.5
        elif avg_saturation > 150:
            return SceneContext.HOME, 0.5
        elif edge_density < 0.05:
            return SceneContext.OUTDOOR, 0.6
        else:
            return SceneContext.UNKNOWN, 0.3


class PromptTemplateLibrary:
    """Library of context-aware prompt templates."""

    def __init__(self):
        self._build_template_library()
        self._build_style_library()
        self._build_negative_prompts()

    def _build_template_library(self):
        """Build comprehensive prompt templates for different contexts."""
        self.templates = {
            # Office contexts
            (SceneContext.OFFICE, ObjectType.PERSON): [
                "professional silhouette in business attire, abstract figure, corporate setting",
                "anonymous office worker figure, geometric representation, workplace appropriate",
                "stylized business person outline, privacy-safe professional avatar"
            ],
            (SceneContext.OFFICE, ObjectType.SCREEN): [
                "generic monitor displaying abstract dashboard, no readable text",
                "computer screen with geometric patterns, corporate color scheme",
                "abstract data visualization on display, privacy-safe interface"
            ],
            (SceneContext.OFFICE, ObjectType.DOCUMENT): [
                "blurred document with lorem ipsum text, professional formatting",
                "abstract paper with unreadable lines, business document style",
                "generic report template, no sensitive information visible"
            ],

            # Medical contexts
            (SceneContext.MEDICAL, ObjectType.PERSON): [
                "medical mannequin figure, anatomically neutral, clinical setting",
                "healthcare avatar in medical attire, no identifying features",
                "abstract medical professional silhouette, HIPAA compliant"
            ],
            (SceneContext.MEDICAL, ObjectType.SCREEN): [
                "medical monitor with generic vital signs display, no patient data",
                "abstract medical interface, privacy-compliant visualization",
                "healthcare dashboard with synthetic data, clinical color scheme"
            ],

            # Educational contexts
            (SceneContext.EDUCATIONAL, ObjectType.PERSON): [
                "educational avatar figure, classroom appropriate, no identity",
                "student silhouette in learning environment, anonymous representation",
                "abstract figure in academic setting, privacy-preserving style"
            ],
            (SceneContext.EDUCATIONAL, ObjectType.WHITEBOARD): [
                "abstract educational content on board, no readable text",
                "geometric shapes and diagrams on whiteboard, teaching visual",
                "blurred academic content, educational patterns only"
            ],

            # Conference/Video call contexts
            (SceneContext.CONFERENCE, ObjectType.PERSON): [
                "professional avatar for video conference, business appropriate",
                "virtual meeting participant silhouette, corporate background",
                "anonymous conference attendee, professional virtual presence"
            ],
            (SceneContext.CONFERENCE, ObjectType.FACE): [
                "professional avatar face, meeting appropriate, no identity",
                "stylized video call participant, business avatar style",
                "generic professional headshot, virtual meeting safe"
            ],

            # Home contexts
            (SceneContext.HOME, ObjectType.PERSON): [
                "casual figure silhouette, home environment, family-safe",
                "residential avatar, comfortable setting, no identifying features",
                "abstract person in domestic space, privacy preserved"
            ],

            # Outdoor contexts
            (SceneContext.OUTDOOR, ObjectType.PERSON): [
                "shadow figure in natural lighting, outdoor silhouette",
                "abstract person shape in environment, nature setting",
                "anonymous figure outdoors, artistic representation"
            ],
            (SceneContext.OUTDOOR, ObjectType.LICENSE_PLATE): [
                "generic license plate with abstract characters, no real numbers",
                "blurred vehicle identification, privacy-safe plate",
                "abstract rectangular shape in place of license plate"
            ]
        }

        # Default templates for unknown contexts
        self.default_templates = {
            ObjectType.PERSON: "anonymous human figure, privacy-preserving silhouette, no identifying features",
            ObjectType.FACE: "abstract face representation, no identity, privacy-safe avatar",
            ObjectType.SCREEN: "generic display with abstract content, no readable information",
            ObjectType.DOCUMENT: "blurred paper with unreadable text, document shape only",
            ObjectType.PHONE: "generic mobile device, abstract screen, no content visible",
            ObjectType.LAPTOP: "generic laptop computer, closed or abstract screen"
        }

    def _build_style_library(self):
        """Build style modifiers for different tones."""
        self.style_modifiers = {
            EmotionalTone.PROFESSIONAL: "corporate style, business appropriate, clean lines",
            EmotionalTone.CASUAL: "relaxed style, friendly appearance, approachable",
            EmotionalTone.FORMAL: "formal presentation, serious tone, official style",
            EmotionalTone.CREATIVE: "artistic interpretation, creative style, unique aesthetic",
            EmotionalTone.TECHNICAL: "technical diagram style, schematic appearance",
            EmotionalTone.FRIENDLY: "warm and welcoming, soft edges, approachable",
            EmotionalTone.SERIOUS: "serious tone, minimal style, focused",
            EmotionalTone.NEUTRAL: "neutral style, balanced appearance, standard"
        }

    def _build_negative_prompts(self):
        """Build negative prompts to avoid unwanted features."""
        self.negative_prompts = {
            ObjectType.PERSON: "face, facial features, identity, recognizable person, real human",
            ObjectType.FACE: "real face, actual person, identity, recognizable features, photograph",
            ObjectType.SCREEN: "readable text, real data, sensitive information, actual content",
            ObjectType.DOCUMENT: "readable text, real information, sensitive data, actual document",
            ObjectType.PHONE: "screen content, readable display, personal information",
            ObjectType.LAPTOP: "screen content, readable display, real data, sensitive information"
        }

        # Universal negative terms
        self.universal_negative = "nsfw, inappropriate, offensive, disturbing, scary, violent"

    def get_template(self, context: SceneContext, obj_type: ObjectType,
                     tone: EmotionalTone = EmotionalTone.NEUTRAL) -> str:
        """Get appropriate prompt template for context."""
        # Try specific context-object combination
        key = (context, obj_type)
        if key in self.templates:
            templates = self.templates[key]
            # Rotate through templates for variety
            template = templates[hash(str(key)) % len(templates)]
        else:
            # Fall back to default
            template = self.default_templates.get(
                obj_type,
                "abstract shape, privacy-safe representation"
            )

        # Add style modifier
        style = self.style_modifiers.get(tone, "")
        if style:
            template = f"{template}, {style}"

        return template

    def get_negative_prompt(self, obj_type: ObjectType) -> str:
        """Get negative prompt for object type."""
        specific = self.negative_prompts.get(obj_type, "")
        return f"{specific}, {self.universal_negative}"


class ContextualPromptEngine:
    """Main engine that generates contextual prompts for privacy protection."""

    def __init__(self):
        print("Initializing Contextual Prompt Engine...")
        self.scene_analyzer = SceneAnalyzer()
        self.template_library = PromptTemplateLibrary()

        # Memory for temporal consistency
        self.context_memory = {}
        self.style_memory = {}
        self.object_history = defaultdict(list)

        # Cache for performance
        self.prompt_cache = {}
        self.cache_size = 100

        print("Contextual Prompt Engine ready!")

    def generate_prompt(self,
                       frame: np.ndarray,
                       region: Dict,
                       tracking_id: Optional[int] = None) -> ContextualPrompt:
        """Generate contextual prompt for a region."""

        # Analyze scene context
        scene_context, confidence = self.scene_analyzer.analyze_scene(frame)

        # Determine object type
        obj_type = self._classify_object(region)

        # Determine emotional tone
        tone = self._determine_tone(scene_context, confidence)

        # Check if we have history for this object
        if tracking_id and tracking_id in self.style_memory:
            # Use consistent style for tracked object
            base_style = self.style_memory[tracking_id]
            prompt = self._evolve_prompt(base_style, scene_context, obj_type)
        else:
            # Generate new prompt
            prompt = self._create_new_prompt(scene_context, obj_type, tone)

            # Store for consistency
            if tracking_id:
                self.style_memory[tracking_id] = prompt

        return prompt

    def _classify_object(self, region: Dict) -> ObjectType:
        """Classify the object type from region data."""
        # Check if classification is provided
        if 'class' in region:
            class_name = region['class']
            if isinstance(class_name, str):
                class_lower = class_name.lower()
                if 'person' in class_lower:
                    return ObjectType.PERSON
                elif 'face' in class_lower:
                    return ObjectType.FACE
                elif 'laptop' in class_lower or 'computer' in class_lower:
                    return ObjectType.LAPTOP
                elif 'phone' in class_lower:
                    return ObjectType.PHONE
                elif 'screen' in class_lower or 'monitor' in class_lower or 'tv' in class_lower:
                    return ObjectType.SCREEN
                elif 'document' in class_lower or 'paper' in class_lower:
                    return ObjectType.DOCUMENT

        # Check region size and aspect ratio for hints
        if 'bbox' in region:
            x1, y1, x2, y2 = region['bbox']
            w, h = x2 - x1, y2 - y1
            aspect = w / h if h > 0 else 1

            if 0.4 < aspect < 0.7 and h > w:
                # Tall and narrow - likely a person
                return ObjectType.PERSON
            elif 1.3 < aspect < 1.8:
                # Wide - likely a screen
                return ObjectType.SCREEN

        return ObjectType.UNKNOWN

    def _determine_tone(self, context: SceneContext, confidence: float) -> EmotionalTone:
        """Determine appropriate emotional tone for the context."""
        tone_mapping = {
            SceneContext.OFFICE: EmotionalTone.PROFESSIONAL,
            SceneContext.MEDICAL: EmotionalTone.SERIOUS,
            SceneContext.EDUCATIONAL: EmotionalTone.NEUTRAL,
            SceneContext.HOME: EmotionalTone.CASUAL,
            SceneContext.OUTDOOR: EmotionalTone.CASUAL,
            SceneContext.CONFERENCE: EmotionalTone.PROFESSIONAL,
            SceneContext.RETAIL: EmotionalTone.FRIENDLY,
            SceneContext.INDUSTRIAL: EmotionalTone.TECHNICAL
        }

        # Adjust based on confidence
        if confidence < 0.5:
            return EmotionalTone.NEUTRAL

        return tone_mapping.get(context, EmotionalTone.NEUTRAL)

    def _create_new_prompt(self,
                          context: SceneContext,
                          obj_type: ObjectType,
                          tone: EmotionalTone) -> ContextualPrompt:
        """Create a new contextual prompt."""

        # Get base template
        base_prompt = self.template_library.get_template(context, obj_type, tone)

        # Get negative prompt
        negative_prompt = self.template_library.get_negative_prompt(obj_type)

        # Add quality boosters
        quality_tokens = [
            "high quality",
            "professional",
            "clean",
            "well-composed"
        ]

        # Add context-specific tokens
        context_tokens = self._get_context_tokens(context)

        # Determine preservation hints
        preservation_hints = self._get_preservation_hints(obj_type, context)

        # Adjust strength based on context
        strength = self._calculate_strength(context, obj_type)

        # Adjust guidance scale
        guidance = self._calculate_guidance(context, obj_type)

        return ContextualPrompt(
            base_prompt=f"{base_prompt}, {', '.join(quality_tokens)}",
            negative_prompt=negative_prompt,
            style_tokens=context_tokens,
            preservation_hints=preservation_hints,
            strength=strength,
            guidance_scale=guidance
        )

    def _evolve_prompt(self,
                       base_style: ContextualPrompt,
                       context: SceneContext,
                       obj_type: ObjectType) -> ContextualPrompt:
        """Evolve an existing prompt for consistency."""
        # Keep base style but adjust for current context
        evolved = ContextualPrompt(
            base_prompt=base_style.base_prompt,
            negative_prompt=base_style.negative_prompt,
            style_tokens=base_style.style_tokens.copy(),
            preservation_hints=base_style.preservation_hints.copy(),
            strength=base_style.strength,
            guidance_scale=base_style.guidance_scale
        )

        # Add slight variations to prevent exact copies
        variation_tokens = [
            "slightly different angle",
            "subtle variation",
            "consistent style"
        ]
        evolved.style_tokens.append(np.random.choice(variation_tokens))

        return evolved

    def _get_context_tokens(self, context: SceneContext) -> List[str]:
        """Get context-specific style tokens."""
        tokens = {
            SceneContext.OFFICE: ["corporate", "professional", "business"],
            SceneContext.MEDICAL: ["clinical", "sterile", "medical"],
            SceneContext.EDUCATIONAL: ["academic", "educational", "learning"],
            SceneContext.HOME: ["domestic", "comfortable", "residential"],
            SceneContext.OUTDOOR: ["natural", "environmental", "outdoor"],
            SceneContext.CONFERENCE: ["virtual", "digital", "meeting"],
            SceneContext.RETAIL: ["commercial", "retail", "shopping"],
            SceneContext.INDUSTRIAL: ["industrial", "technical", "mechanical"]
        }
        return tokens.get(context, ["neutral", "standard"])

    def _get_preservation_hints(self, obj_type: ObjectType, context: SceneContext) -> Dict:
        """Determine what aspects to preserve in generation."""
        hints = {
            "preserve_pose": obj_type in [ObjectType.PERSON, ObjectType.FACE],
            "preserve_size": True,
            "preserve_position": True,
            "preserve_lighting": context != SceneContext.UNKNOWN,
            "preserve_color_tone": context in [SceneContext.OFFICE, SceneContext.MEDICAL],
            "preserve_gesture": obj_type == ObjectType.PERSON and context == SceneContext.CONFERENCE
        }
        return hints

    def _calculate_strength(self, context: SceneContext, obj_type: ObjectType) -> float:
        """Calculate inpainting strength based on context."""
        # Higher strength = more change from original
        if context == SceneContext.MEDICAL:
            return 0.95  # Maximum privacy for medical
        elif context == SceneContext.OFFICE and obj_type == ObjectType.DOCUMENT:
            return 0.9  # High privacy for office documents
        elif context == SceneContext.HOME:
            return 0.7  # Moderate for home
        elif context == SceneContext.OUTDOOR:
            return 0.6  # Less strict for outdoor
        else:
            return 0.8  # Default

    def _calculate_guidance(self, context: SceneContext, obj_type: ObjectType) -> float:
        """Calculate guidance scale for diffusion."""
        # Higher guidance = more adherence to prompt
        if context in [SceneContext.MEDICAL, SceneContext.OFFICE]:
            return 8.5  # Strict adherence for professional settings
        elif context == SceneContext.EDUCATIONAL:
            return 7.5  # Moderate
        elif context in [SceneContext.HOME, SceneContext.OUTDOOR]:
            return 6.5  # More creative freedom
        else:
            return 7.5  # Default

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "prompt_cache_size": len(self.prompt_cache),
            "style_memory_size": len(self.style_memory),
            "context_memory_size": len(self.context_memory)
        }

    def clear_memory(self):
        """Clear temporal memory for new session."""
        self.context_memory.clear()
        self.style_memory.clear()
        self.object_history.clear()


def test_contextual_engine():
    """Test the contextual prompt engine."""
    print("\n" + "="*80)
    print("TESTING CONTEXTUAL PROMPT ENGINE")
    print("="*80)

    # Initialize engine
    engine = ContextualPromptEngine()

    # Create test scenarios
    test_scenarios = [
        {
            "name": "Office Meeting",
            "frame": np.ones((720, 1280, 3), dtype=np.uint8) * 200,  # Bright office
            "region": {"bbox": [100, 100, 300, 500], "class": "person"},
            "expected_context": SceneContext.OFFICE
        },
        {
            "name": "Medical Consultation",
            "frame": np.ones((720, 1280, 3), dtype=np.uint8) * 240,  # Very bright medical
            "region": {"bbox": [200, 150, 400, 450], "class": "person"},
            "expected_context": SceneContext.MEDICAL
        },
        {
            "name": "Video Conference",
            "frame": np.ones((720, 1280, 3), dtype=np.uint8) * 180,  # Medium brightness
            "region": {"bbox": [300, 200, 500, 400], "class": "face"},
            "expected_context": SceneContext.CONFERENCE
        },
        {
            "name": "Outdoor Scene",
            "frame": np.random.randint(100, 200, (720, 1280, 3), dtype=np.uint8),  # Varied outdoor
            "region": {"bbox": [400, 300, 600, 700], "class": "person"},
            "expected_context": SceneContext.OUTDOOR
        }
    ]

    print("\nTesting different scenarios:\n")

    for i, scenario in enumerate(test_scenarios):
        print(f"{i+1}. {scenario['name']}:")
        print("-" * 40)

        # Generate prompt
        prompt = engine.generate_prompt(
            scenario["frame"],
            scenario["region"],
            tracking_id=i
        )

        print(f"Base Prompt: {prompt.base_prompt[:100]}...")
        print(f"Negative: {prompt.negative_prompt[:80]}...")
        print(f"Style Tokens: {prompt.style_tokens}")
        print(f"Strength: {prompt.strength:.2f}")
        print(f"Guidance: {prompt.guidance_scale:.1f}")
        print(f"Preserve Pose: {prompt.preservation_hints.get('preserve_pose', False)}")
        print()

    # Test temporal consistency
    print("\nTesting Temporal Consistency:")
    print("-" * 40)

    # Generate multiple prompts for same tracked object
    for frame_num in range(3):
        prompt = engine.generate_prompt(
            test_scenarios[0]["frame"],
            test_scenarios[0]["region"],
            tracking_id=999  # Same ID for consistency
        )
        print(f"Frame {frame_num}: {prompt.base_prompt[:50]}...")

    print("\n" + "="*80)
    print("✅ Contextual Prompt Engine Test Complete!")
    print("="*80)

    # Show cache stats
    stats = engine.get_cache_stats()
    print(f"\nCache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_contextual_engine()