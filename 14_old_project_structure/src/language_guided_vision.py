"""
Language-Guided Vision System for RealityGuard
Implements natural language control for privacy filtering in AR/VR
Aligns with Meta's preference for language-guided vision systems
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import re
from enum import Enum

# Try importing CLIP for vision-language alignment
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")


class IntentType(Enum):
    """Types of language-guided intents"""
    HIDE_OBJECT = "hide_object"
    BLUR_PERSON = "blur_person"
    PROTECT_SCREEN = "protect_screen"
    FILTER_TEXT = "filter_text"
    CUSTOM_RULE = "custom_rule"
    QUERY = "query"


@dataclass
class LanguageCommand:
    """Represents a parsed language command"""
    raw_text: str
    intent: IntentType
    targets: List[str]
    modifiers: Dict[str, Any]
    confidence: float


@dataclass
class VisionLanguageMatch:
    """Represents a match between language description and visual element"""
    object_bbox: Tuple[int, int, int, int]
    description: str
    confidence: float
    action: str


class LanguageGuidedVision:
    """
    Language-guided vision system for natural language control of privacy filtering

    This demonstrates Meta's preferred qualification: language-guided vision systems
    """

    def __init__(self, use_clip: bool = True):
        """Initialize language-guided vision system

        Args:
            use_clip: Whether to use CLIP for vision-language alignment
        """
        self.use_clip = use_clip and CLIP_AVAILABLE

        if self.use_clip:
            self._init_clip()

        # Intent patterns for natural language understanding
        self.intent_patterns = {
            IntentType.HIDE_OBJECT: [
                r"hide (?:the )?(\w+)",
                r"remove (?:the )?(\w+)",
                r"don't show (?:the )?(\w+)",
                r"block (?:the )?(\w+)"
            ],
            IntentType.BLUR_PERSON: [
                r"blur (?:the )?(\w+)",
                r"blur (?:all )?people",
                r"protect (?:the )?person",
                r"anonymize (?:the )?(\w+)"
            ],
            IntentType.PROTECT_SCREEN: [
                r"hide (?:all )?screens",
                r"protect (?:my )?screen",
                r"blur (?:the )?monitor",
                r"hide (?:the )?display"
            ],
            IntentType.FILTER_TEXT: [
                r"hide (?:all )?text",
                r"blur (?:any )?text",
                r"protect (?:written )?information",
                r"remove (?:all )?words"
            ],
            IntentType.CUSTOM_RULE: [
                r"when (?:I'm |i'm )?(\w+) then (\w+)",
                r"if (\w+) is visible then (\w+)",
                r"always (\w+) when (\w+)"
            ]
        }

        # Object detection labels for language grounding
        self.object_vocabulary = {
            'person': ['person', 'people', 'human', 'face', 'user'],
            'screen': ['screen', 'monitor', 'display', 'laptop', 'computer', 'phone'],
            'text': ['text', 'words', 'writing', 'document', 'paper'],
            'credit_card': ['credit card', 'card', 'payment card', 'bank card'],
            'keyboard': ['keyboard', 'keys', 'typing device'],
            'whiteboard': ['whiteboard', 'board', 'presentation'],
            'badge': ['badge', 'id', 'identification', 'pass']
        }

        # Privacy rules database
        self.privacy_rules = []

    def _init_clip(self):
        """Initialize CLIP model for vision-language alignment"""
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
            self.clip_model.eval()
            print("CLIP model loaded for vision-language alignment")
        except Exception as e:
            print(f"Failed to load CLIP: {e}")
            self.use_clip = False

    def parse_command(self, text: str) -> LanguageCommand:
        """Parse natural language command into structured intent

        Args:
            text: Natural language command

        Returns:
            Parsed language command
        """
        text_lower = text.lower()

        # Match against intent patterns
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    targets = [g for g in match.groups() if g]

                    # Extract modifiers
                    modifiers = self._extract_modifiers(text_lower)

                    return LanguageCommand(
                        raw_text=text,
                        intent=intent_type,
                        targets=targets,
                        modifiers=modifiers,
                        confidence=0.9
                    )

        # Default to query if no pattern matches
        return LanguageCommand(
            raw_text=text,
            intent=IntentType.QUERY,
            targets=[],
            modifiers={},
            confidence=0.5
        )

    def _extract_modifiers(self, text: str) -> Dict[str, Any]:
        """Extract modifiers from text (e.g., except, only, always)

        Args:
            text: Input text

        Returns:
            Dictionary of modifiers
        """
        modifiers = {}

        # Check for exceptions
        if 'except' in text:
            except_match = re.search(r'except (?:for )?(\w+)', text)
            if except_match:
                modifiers['except'] = except_match.group(1)

        # Check for specific people
        if 'only' in text:
            only_match = re.search(r'only (\w+)', text)
            if only_match:
                modifiers['only'] = only_match.group(1)

        # Check for temporal conditions
        if 'when' in text or 'during' in text:
            modifiers['conditional'] = True

        # Check for intensity
        if 'heavily' in text or 'strongly' in text:
            modifiers['intensity'] = 'high'
        elif 'lightly' in text or 'slightly' in text:
            modifiers['intensity'] = 'low'

        return modifiers

    def ground_language_to_vision(self,
                                 command: LanguageCommand,
                                 frame: np.ndarray,
                                 detections: List[Dict]) -> List[VisionLanguageMatch]:
        """Ground language command to visual elements in frame

        Args:
            command: Parsed language command
            frame: Current frame
            detections: List of detected objects with bboxes

        Returns:
            List of vision-language matches
        """
        matches = []

        if self.use_clip and CLIP_AVAILABLE:
            # Use CLIP for sophisticated matching
            matches = self._clip_matching(command, frame, detections)
        else:
            # Use rule-based matching
            matches = self._rule_based_matching(command, detections)

        return matches

    def _clip_matching(self,
                      command: LanguageCommand,
                      frame: np.ndarray,
                      detections: List[Dict]) -> List[VisionLanguageMatch]:
        """Use CLIP for vision-language matching

        Args:
            command: Language command
            frame: Current frame
            detections: Detected objects

        Returns:
            Vision-language matches
        """
        matches = []

        if not self.use_clip:
            return matches

        try:
            import torch

            # Prepare text descriptions
            text_descriptions = []
            for target in command.targets:
                text_descriptions.append(f"a photo of a {target}")

            if not text_descriptions:
                text_descriptions = [f"a photo of a {command.raw_text}"]

            # Encode text
            text_tokens = clip.tokenize(text_descriptions).cpu()

            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            # Process each detection
            for det in detections:
                x, y, w, h = det['bbox']

                # Extract region
                region = frame[y:y+h, x:x+w]
                if region.size == 0:
                    continue

                # Preprocess for CLIP
                region_pil = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                region_tensor = self.clip_preprocess(region_pil).unsqueeze(0).cpu()

                # Encode image region
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(region_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                # Calculate similarity
                similarity = (image_features @ text_features.T).squeeze()
                max_similarity = similarity.max().item()

                if max_similarity > 0.25:  # Threshold for matching
                    best_idx = similarity.argmax().item()

                    matches.append(VisionLanguageMatch(
                        object_bbox=(x, y, w, h),
                        description=command.targets[best_idx] if command.targets else "object",
                        confidence=max_similarity,
                        action=self._get_action_for_intent(command.intent)
                    ))

        except Exception as e:
            print(f"CLIP matching error: {e}")

        return matches

    def _rule_based_matching(self,
                           command: LanguageCommand,
                           detections: List[Dict]) -> List[VisionLanguageMatch]:
        """Rule-based matching when CLIP is not available

        Args:
            command: Language command
            detections: Detected objects

        Returns:
            Vision-language matches
        """
        matches = []

        for det in detections:
            category = det.get('category', 'unknown')

            # Check if detection matches any target
            for target in command.targets:
                if self._matches_vocabulary(target, category):
                    matches.append(VisionLanguageMatch(
                        object_bbox=det['bbox'],
                        description=target,
                        confidence=0.8,
                        action=self._get_action_for_intent(command.intent)
                    ))
                    break

        return matches

    def _matches_vocabulary(self, target: str, category: str) -> bool:
        """Check if target matches category using vocabulary

        Args:
            target: Target from language command
            category: Detected object category

        Returns:
            True if matches
        """
        for obj_type, synonyms in self.object_vocabulary.items():
            if target in synonyms and (obj_type == category or category in synonyms):
                return True
        return False

    def _get_action_for_intent(self, intent: IntentType) -> str:
        """Get action for intent type

        Args:
            intent: Intent type

        Returns:
            Action string
        """
        action_map = {
            IntentType.HIDE_OBJECT: "remove",
            IntentType.BLUR_PERSON: "blur",
            IntentType.PROTECT_SCREEN: "pixelate",
            IntentType.FILTER_TEXT: "blur",
            IntentType.CUSTOM_RULE: "custom",
            IntentType.QUERY: "highlight"
        }
        return action_map.get(intent, "blur")

    def add_privacy_rule(self, rule_text: str):
        """Add a privacy rule from natural language

        Args:
            rule_text: Natural language privacy rule
        """
        command = self.parse_command(rule_text)

        self.privacy_rules.append({
            'text': rule_text,
            'command': command,
            'active': True
        })

        print(f"Added privacy rule: {rule_text}")

    def apply_language_guided_filtering(self,
                                       frame: np.ndarray,
                                       command_text: str,
                                       detections: List[Dict]) -> Tuple[np.ndarray, List[VisionLanguageMatch]]:
        """Apply language-guided filtering to frame

        Args:
            frame: Input frame
            command_text: Natural language command
            detections: Detected objects

        Returns:
            Filtered frame and matches
        """
        # Parse command
        command = self.parse_command(command_text)

        # Ground language to vision
        matches = self.ground_language_to_vision(command, frame, detections)

        # Apply filtering based on matches
        output = frame.copy()

        for match in matches:
            x, y, w, h = match.object_bbox

            if match.action == "blur":
                roi = output[y:y+h, x:x+w]
                blurred = cv2.GaussianBlur(roi, (31, 31), 0)
                output[y:y+h, x:x+w] = blurred

            elif match.action == "pixelate":
                roi = output[y:y+h, x:x+w]
                temp = cv2.resize(roi, (w//20, h//20), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                output[y:y+h, x:x+w] = pixelated

            elif match.action == "remove":
                output[y:y+h, x:x+w] = 128  # Gray fill

            elif match.action == "highlight":
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 3)

        return output, matches

    def process_with_context(self,
                           frame: np.ndarray,
                           context: str,
                           detections: List[Dict]) -> np.ndarray:
        """Process frame with contextual understanding

        Args:
            frame: Input frame
            context: Context description (e.g., "I'm in a meeting")
            detections: Detected objects

        Returns:
            Processed frame
        """
        # Generate appropriate command based on context
        context_commands = {
            "meeting": "blur all people except me and hide all screens",
            "public": "protect all personal information and blur faces",
            "recording": "blur all people and hide sensitive information",
            "streaming": "blur everything except my face",
            "working": "hide all screens except mine"
        }

        # Find matching context
        command_text = None
        for key, cmd in context_commands.items():
            if key in context.lower():
                command_text = cmd
                break

        if command_text:
            output, _ = self.apply_language_guided_filtering(frame, command_text, detections)
            return output

        return frame


class LanguageGuidedDemo:
    """Demo for language-guided vision capabilities"""

    @staticmethod
    def demonstrate_capabilities():
        """Demonstrate language-guided vision features"""

        print("\n" + "="*60)
        print("LANGUAGE-GUIDED VISION DEMONSTRATION")
        print("Showcasing Meta's Preferred Qualification")
        print("="*60)

        # Initialize system
        lgv = LanguageGuidedVision(use_clip=False)  # Use rule-based for demo

        # Example commands
        example_commands = [
            "Hide all screens in the room",
            "Blur everyone except me",
            "Protect my credit card information",
            "When I'm in a meeting, blur all faces",
            "Only show my face and blur everything else",
            "Remove all text from the view"
        ]

        print("\nExample Natural Language Commands:")
        print("-" * 40)

        for cmd in example_commands:
            parsed = lgv.parse_command(cmd)
            print(f"\nCommand: '{cmd}'")
            print(f"  Intent: {parsed.intent.value}")
            print(f"  Targets: {parsed.targets}")
            print(f"  Modifiers: {parsed.modifiers}")

        # Simulate detection and matching
        mock_detections = [
            {'bbox': (100, 100, 200, 200), 'category': 'person'},
            {'bbox': (400, 150, 400, 300), 'category': 'screen'},
            {'bbox': (50, 350, 150, 100), 'category': 'text'}
        ]

        # Create test frame
        test_frame = np.ones((600, 800, 3), dtype=np.uint8) * 128

        # Apply language-guided filtering
        print("\n" + "="*60)
        print("Applying Language-Guided Filtering")
        print("-" * 40)

        for cmd in example_commands[:3]:
            output, matches = lgv.apply_language_guided_filtering(
                test_frame, cmd, mock_detections
            )

            print(f"\nCommand: '{cmd}'")
            print(f"Matches found: {len(matches)}")
            for match in matches:
                print(f"  - {match.description} at {match.object_bbox[:2]} "
                     f"(confidence: {match.confidence:.2f}, action: {match.action})")

        print("\n✅ Language-Guided Vision System Ready")
        print("   Perfect for Meta's AR/VR applications!")


if __name__ == "__main__":
    # Run demonstration
    LanguageGuidedDemo.demonstrate_capabilities()