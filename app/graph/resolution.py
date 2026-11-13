"""Global entity resolution with local embeddings and reviewed calibration.

No LLM calls and no downloads. Ambiguous matches block, never become new nodes.
"""
import math
import uuid
from pathlib import Path

from app.evaluation.extraction_benchmark import identity
from app.graph.construction import normalized, resolve


def entity_text(entity):
    return (entity['entity_type'] + ': ' + normalized(entity['canonical_name']) + '. '
            + entity['description'])


def cosine(left, right):
    if len(left) != len(right) or not left:
        raise ValueError('inconsistent embedding dimensions')
    for value in list(left) + list(right):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('embedding must contain finite numbers')
    a = sum(v * v for v in left)
    b = sum(v * v for v in right)
    if not a or not b:
        raise ValueError('zero embedding vector')
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(left, right)) / math.sqrt(a * b)))


class LocalEmbedder:
    def __init__(self, model_path):
        # Explicit local directory avoids silent download/account/network work.
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise ValueError('a locally installed embedding model directory is required')
        files = []
        import hashlib
        for path in sorted(model_path.rglob('*')):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as handle:
                    for part in iter(lambda: handle.read(1024 * 1024), b''):
                        digest.update(part)
                files.append([str(path.relative_to(model_path)), digest.hexdigest()])
        self.fingerprint = identity(files)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(model_path), device='cpu', local_files_only=True, trust_remote_code=False)

    def __call__(self, texts):
        return self.model.encode(texts, batch_size=1, normalize_embeddings=True, show_progress_bar=False).tolist()


def calibrate(pairs, embedder):
    """Find a separating threshold on reviewed pairs, not a guessed constant.

    Both match and nonmatch examples required. Overlap means auto-resolution is
    unsafe on this calibration set: collect better labels/model, do not relax.
    """
    positive = []
    negative = []
    for pair in pairs:
        expected_hash = identity([pair['left'], pair['right']])
        if (pair.get('reviewed') is not True or pair.get('pair_hash') != expected_hash
                or type(pair.get('same_entity')) is not bool):
            raise ValueError('source-bound reviewed identity pairs required')
        if pair['left']['entity_type'] != pair['right']['entity_type']:
            raise ValueError('calibrate within the same entity type; type filtering already separates other types')
        vectors = embedder([entity_text(pair['left']), entity_text(pair['right'])])
        score = cosine(vectors[0], vectors[1])
        (positive if pair['same_entity'] else negative).append(score)
    if not positive or not negative:
        raise ValueError('calibration needs reviewed matches AND nonmatches')
    if max(negative) >= min(positive):
        raise ValueError('no separating threshold on reviewed examples; automatic resolution blocked')
    return {'model_fingerprint': embedder.fingerprint, 'threshold': min(positive),
            'calibration_hash': identity(pairs), 'matches': len(positive), 'nonmatches': len(negative),
            'warning': 'Separates this calibration sample only, not a general identity-accuracy guarantee.'}


def global_resolve(records, registry, embedder, calibration, identity_reviews=None):
    if calibration.get('model_fingerprint') != embedder.fingerprint:
        raise ValueError('embedding model differs from calibrated model')
    threshold = calibration['threshold']
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('invalid calibrated threshold')
    resolved = resolve(records)
    # Snapshot must come from the intended DB before a real write. Never guess
    # that a local empty registry means the graph is empty.
    if len(registry['entities']) > 10000 or len(records) > 1000:
        raise ValueError('small CPU path limited to 10000 registry entities / 1000 chunks; review scale before expanding')
    entities = {e['id']: dict(e) for e in registry['entities']}
    if len(entities) != len(registry['entities']):
        raise ValueError('duplicate canonical IDs in registry')
    for entity in entities.values():
        if entity.get('model_fingerprint') != embedder.fingerprint:
            raise ValueError('registry embedding model differs; explicit reindex required')
    mentions = resolved['mentions']
    overrides = {}
    mention_ids = {m['mention_id'] for m in mentions}
    for review in identity_reviews or []:
        mention_id = review.get('mention_id')
        if review.get('reviewed') is not True or mention_id not in mention_ids or mention_id in overrides:
            raise ValueError('identity review must bind a unique current mention')
        overrides[mention_id] = review['entity_id']
    vectors = embedder([entity_text(m) for m in mentions]) if mentions else []
    if len(vectors) != len(mentions):
        raise ValueError('embedding output count differs from inputs')
    decisions = []
    unresolved = []
    for mention, vector in zip(mentions, vectors):
        candidates = []
        alias_conflicts = []
        mention_names = {normalized(n) for n in [mention['canonical_name']] + mention['aliases']}
        for entity in entities.values():
            if entity['entity_type'] != mention['entity_type']:
                continue
            existing_names = {normalized(n) for n in [entity['canonical_name']] + entity['aliases']}
            if mention_names & existing_names:
                alias_conflicts.append(entity['id'])
            score = cosine(vector, entity['embedding'])
            if score >= threshold:
                candidates.append((entity['id'], score))
        override = overrides.get(mention['mention_id'])
        if override is not None:
            if override != 'new' and (override not in entities or entities[override]['entity_type'] != mention['entity_type']):
                raise ValueError('review selected an unknown or wrong-type canonical entity')
            candidates = [] if override == 'new' else [(override, None)]
            alias_conflicts = []
        if len(candidates) > 1 or (alias_conflicts and (not candidates or any(i != candidates[0][0] for i in alias_conflicts))):
            unresolved.append({'mention_id': mention['mention_id'], 'candidates': candidates,
                               'reason': 'multiple above-threshold identities; no new node allocated'})
            continue
        if candidates:
            entity_id, score = candidates[0]
            decision = 'reuse'
        else:
            entity_id = str(uuid.uuid4())  # persistent registry ID, never LLM-generated
            score = None
            decision = 'new'
            entities[entity_id] = {'id': entity_id, 'canonical_name': mention['canonical_name'],
                'entity_type': mention['entity_type'], 'description': mention['description'],
                'aliases': [], 'embedding': vector, 'model_fingerprint': embedder.fingerprint}
        aliases = set(entities[entity_id]['aliases'])
        aliases.update(mention['aliases'])
        aliases.add(mention['canonical_name'])
        entities[entity_id]['aliases'] = sorted(aliases)
        resolved['mention_to_entity'][mention['mention_id']] = entity_id
        decisions.append({'mention_id': mention['mention_id'], 'entity_id': entity_id,
                          'decision': decision, 'cosine': score})
    if unresolved:
        return {'status': 'Needs identity review', 'unresolved': unresolved,
                'writes_permitted': False, 'registry_version': registry['version']}
    # Rebuild edges with the new mapping, retaining original raw predictions.
    old = resolve(records)
    id_map = {old_id: resolved['mention_to_entity'][old_id] for old_id in old['mention_to_entity']}
    for edge in resolved['relationships']:
        edge['source_id'] = id_map[edge['source_id']]
        edge['target_id'] = id_map[edge['target_id']]
    resolved.update(status='Resolved with calibrated similarity', canonical_entities=list(entities.values()),
                    decisions=decisions, calibration=calibration, registry_version=registry['version'],
                    registry_target=registry.get('target'), extraction_hash=identity(records),
                    requires_semantic_review=True)
    return resolved
