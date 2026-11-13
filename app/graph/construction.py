"""Source-local identities, reviewable resolution candidates and offline Cypher.

No fuzzy match is an identity proof. No database connection or writes here.
"""
import re

from app.evaluation.extraction_benchmark import identity


def normalized(name):
    return ' '.join(name.casefold().split())


def resolve(records, reviewed_merges=None):
    """Distinct source-local mentions until an exact mention-set review merges.

    reviewed_merges: [{reviewed: true, mention_ids: [...]}]. IDs bind the exact
    source hash and extracted entity. Candidate matching never merges by itself.
    """
    mentions = {}
    endpoints = {}
    edges = []
    for record in records:
        if not record.get('accepted'):
            continue
        for entity in record['entities']:
            mention_id = identity([record['source_hash'], record['chunk_id'], entity])
            if mention_id in mentions:
                raise ValueError('duplicate mention input')
            mentions[mention_id] = dict(entity, mention_id=mention_id)
            endpoint = (record['source_hash'], record['chunk_id'], entity['canonical_name'])
            if endpoint in endpoints:
                raise ValueError('duplicate local endpoint')
            endpoints[endpoint] = mention_id
        for edge in record['relationships']:
            edges.append((record, edge))
    mappings = {mention_id: mention_id for mention_id in mentions}
    used = set()
    for review in reviewed_merges or []:
        ids = review.get('mention_ids', [])
        if review.get('reviewed') is not True or len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError('merge needs an explicit reviewed unique mention set')
        if any(i not in mentions or i in used for i in ids):
            raise ValueError('unknown or overlapping reviewed identity set')
        if len({mentions[i]['entity_type'] for i in ids}) != 1:
            raise ValueError('cross-type merge requires ontology review, not automatic identity merge')
        resolved_id = identity(sorted(ids))
        for mention_id in ids:
            mappings[mention_id] = resolved_id
            used.add(mention_id)
    candidates = []
    ids = sorted(mentions)
    # Token index avoids comparing every pair in the corpus.
    index = {}
    pairs = set()
    for mention_id in ids:
        mention = mentions[mention_id]
        names = [mention['canonical_name']] + mention['aliases']
        for name in set(normalized(n) for n in names):
            key = (mention['entity_type'], name)
            for other in index.get(key, []):
                pair = tuple(sorted((mention_id, other)))
                if pair not in pairs and mappings[mention_id] != mappings[other]:
                    candidates.append({'mention_ids': list(pair), 'reason': 'same normalized name or alias',
                                       'auto_merge': False})
                    pairs.add(pair)
            index.setdefault(key, []).append(mention_id)
    resolved_edges = []
    for record, edge in edges:
        source = endpoints[(record['source_hash'], record['chunk_id'], edge['source_entity'])]
        target = endpoints[(record['source_hash'], record['chunk_id'], edge['target_entity'])]
        resolved_edges.append({'source_id': mappings[source], 'target_id': mappings[target],
                               'edge': edge, 'source_hash': record['source_hash'],
                               'prediction_hash': identity(edge)})
    return {'mentions': list(mentions.values()), 'mention_to_entity': mappings,
            'relationships': resolved_edges, 'resolution_candidates': candidates,
            'policy': 'No similarity-based automatic merges; source-local identity until review.'}


def graph_plan(records, reviewed_merges, evidence_reviews, global_resolution=None):
    """Parameterized MERGEs, only from source-bound positive semantic reviews.

    Each relationship retains its own citation; repeated claims never overwrite
    different source evidence. Entity review covers aliases and description too.
    """
    resolved = resolve(records, reviewed_merges)
    if global_resolution is not None:
        if (global_resolution.get('extraction_hash') != identity(records)
                or global_resolution.get('status') != 'Resolved with calibrated similarity'
                or set(global_resolution['mention_to_entity']) != set(resolved['mention_to_entity'])):
            raise ValueError('global resolution is incomplete or stale')
        mapping = global_resolution['mention_to_entity']
        for relation in resolved['relationships']:
            relation['source_id'] = mapping[relation['source_id']]
            relation['target_id'] = mapping[relation['target_id']]
        resolved['mention_to_entity'] = mapping
    statements = [
        {'query': 'CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE', 'parameters': {}},
        {'query': 'CREATE CONSTRAINT mention_id IF NOT EXISTS FOR (n:Mention) REQUIRE n.id IS UNIQUE', 'parameters': {}},
        {'query': 'CREATE CONSTRAINT citation_id IF NOT EXISTS FOR (n:Citation) REQUIRE n.id IS UNIQUE', 'parameters': {}},
    ]
    if global_resolution is not None:
        canonical = {e['id']: e for e in global_resolution['canonical_entities']}
        if not set(resolved['mention_to_entity'].values()) <= set(canonical):
            raise ValueError('missing canonical registry nodes')
        for entity in canonical.values():
            statements.append({'query': 'MERGE (e:Entity {id: $entity_id}) SET e += $props',
                               'parameters': {'entity_id': entity['id'], 'props': entity}})
    entities = {}
    for record in records:
        if not record.get('accepted'):
            continue
        source_reviews = evidence_reviews.get(record['chunk_id'], {})
        for entity in record['entities']:
            review = source_reviews.get(identity(entity), {})
            if (review.get('reviewed') is not True or review.get('supported') is not True
                    or review.get('source_hash') != record['source_hash']):
                raise ValueError('every graph entity needs source-bound semantic review')
            mention_id = identity([record['source_hash'], record['chunk_id'], entity])
            entities[mention_id] = dict(entity, source_hash=record['source_hash'])
    if set(entities) != set(resolved['mention_to_entity']):
        raise ValueError('resolution does not match the reviewed extraction')
    for mention_id, entity in entities.items():
        props = dict(entity)
        props['id'] = mention_id
        entity_id = resolved['mention_to_entity'][mention_id]
        statements.append({'query': 'MERGE (e:Entity {id: $entity_id}) MERGE (m:Mention {id: $mention_id}) '
                            'SET m += $props MERGE (m)-[:REFERS_TO]->(e)',
                           'parameters': {'entity_id': entity_id, 'mention_id': mention_id, 'props': props}})
    for relation in resolved['relationships']:
        edge = relation['edge']
        review = evidence_reviews.get(edge['source_chunk_id'], {}).get(relation['prediction_hash'], {})
        if (review.get('reviewed') is not True or review.get('supported') is not True
                or review.get('source_hash') != relation['source_hash']):
            raise ValueError('every graph edge needs source-bound semantic evidence review')
        relation_type = edge['relationship_type']
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*', relation_type):
            raise ValueError('unsafe relationship identifier')
        citation_id = identity(relation)
        props = dict(edge, source_hash=relation['source_hash'])
        statements.append({'query': 'MATCH (s:Entity {id: $source_id}), (t:Entity {id: $target_id}) '
                            'MERGE (s)-[r:' + relation_type + ' {id: $citation_id}]->(t) SET r += $props '
                            'MERGE (c:Citation {id: $citation_id}) SET c += $props '
                            'MERGE (s)-[:CLAIM_SOURCE]->(c) MERGE (c)-[:CLAIM_TARGET]->(t)',
                           'parameters': {'source_id': relation['source_id'], 'target_id': relation['target_id'],
                                          'citation_id': citation_id, 'props': props}})
    return {'statements': statements, 'writes_performed': False,
            'registry_version': global_resolution.get('registry_version') if global_resolution else None,
            'registry_target': global_resolution.get('registry_target') if global_resolution else None,
            'representation': 'Typed directional edges with citation IDs and evidence; source-local mentions preserved.'}
