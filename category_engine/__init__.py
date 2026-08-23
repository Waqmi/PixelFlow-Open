"""Category Template Engine public API."""

from .category_classifier import (
    AICategoryProvider,
    CATEGORY_IDS,
    CategoryDecision,
    RuleCategoryClassifier,
)
from .category_template_manager import CategoryTemplate, CategoryTemplateManager
from .composition_engine import compose_with_template
from .source_type_classifier import (
    AISourceTypeProvider,
    SOURCE_TYPE_IDS,
    RuleSourceTypeClassifier,
    SourceTypeDecision,
)

__all__ = [
    "AICategoryProvider",
    "CATEGORY_IDS",
    "CategoryDecision",
    "CategoryTemplate",
    "CategoryTemplateManager",
    "RuleCategoryClassifier",
    "compose_with_template",
    "AISourceTypeProvider",
    "SOURCE_TYPE_IDS",
    "RuleSourceTypeClassifier",
    "SourceTypeDecision",
]
