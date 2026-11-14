"""Expanded extraction contract for review, separate from resolution and writes.

No model calls or database writes happen here. A model's
canonical_name is only its source-local name proposal, not a resolved graph ID.
Validation proves shape/ontology/provenance/quote presence, not semantic truth.
"""
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, create_model

def squash_spaces(text):
    return ' '.join(text.split())


CONTRACT_VERSION = 'extraction-contract-v3-frozen-types'
NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class EntityExtraction(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    canonical_name: NonemptyText
    entity_type: NonemptyText
    aliases: list[NonemptyText]
    description: NonemptyText
    source_chunk_id: NonemptyText
    confidence: Confidence
    evidence: NonemptyText


class RelationshipExtraction(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    source_entity: NonemptyText
    target_entity: NonemptyText
    relationship_type: NonemptyText
    source_chunk_id: NonemptyText
    confidence: Confidence
    evidence: NonemptyText


class ExtractionPayload(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    entities: list[EntityExtraction]
    relationships: list[RelationshipExtraction]


def check_source_contract(payload, chunk, ontology):
    """Return mechanical problems, never a factual-accuracy verdict.

    Semantic support of entity identity, alias equivalence, descriptions,
    confidence and relationship direction still needs independent source review.
    No alias is globally merged here, no graph writes occur, and no repair runs.
    """
    if not isinstance(payload, ExtractionPayload):
        raise TypeError('payload must pass the common Pydantic contract first')
    chunk_id = chunk['chunk_id']
    source = squash_spaces(chunk['text'])
    problems = []
    names = {}
    for entity in payload.entities:
        if entity.source_chunk_id != chunk_id:
            problems.append('entity has the wrong source chunk ID')
        if entity.entity_type not in ontology.entity_types:
            problems.append('entity type is outside the fixed ontology')
        if entity.canonical_name in names:
            problems.append('duplicate entity name; do not resolve identities in extraction validation')
        names[entity.canonical_name] = entity.entity_type
        if squash_spaces(entity.evidence) not in source:
            problems.append('entity evidence is absent from the original chunk')
        if len(entity.aliases) != len(set(entity.aliases)):
            problems.append('duplicate aliases in one entity')
    for edge in payload.relationships:
        if edge.source_chunk_id != chunk_id:
            problems.append('relationship has the wrong source chunk ID')
        if squash_spaces(edge.evidence) not in source:
            problems.append('relationship evidence is absent from the original chunk')
        if edge.source_entity not in names or edge.target_entity not in names:
            problems.append('relationship endpoint is absent from extracted entities')
        elif not ontology.allows(edge.relationship_type, names[edge.source_entity], names[edge.target_entity]):
            problems.append('relationship direction or type is outside the fixed ontology')
    return {'contract_version': CONTRACT_VERSION, 'mechanical_checks_pass': not problems,
            'problems': problems, 'semantic_evidence_status': 'Not reviewed',
            'global_entity_resolution_status': 'Not performed',
            'graph_write_permitted': False}


@lru_cache(maxsize=32)
def frozen_contract(entity_types, relationship_types):
    """One Literal-bound model from the existing immutable ontology, no second list.

    Native JSON schema and local validation share these same fixed type names.
    Endpoint direction and source support still need the separate checks above.
    """
    if not entity_types or not relationship_types:
        raise ValueError('nonempty frozen ontology required')
    entity = create_model('EntityExtraction', __base__=EntityExtraction,
                          entity_type=(Literal[entity_types], ...))
    relation = create_model('RelationshipExtraction', __base__=RelationshipExtraction,
                            relationship_type=(Literal[relationship_types], ...))
    return create_model('ExtractionPayload', __base__=ExtractionPayload,
                        entities=(list[entity], ...), relationships=(list[relation], ...))


def ontology_contract(ontology):
    return frozen_contract(tuple(ontology.entity_types), tuple(sorted(ontology.relations)))
