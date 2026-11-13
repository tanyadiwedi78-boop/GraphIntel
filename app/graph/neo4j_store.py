"""Explicit target, optimistic registry version, one non-retried data transaction."""
from app.graph.construction import graph_plan


def read_registry(driver, database, target):
    if not database or not target:
        raise ValueError('explicit database and target identity required')
    with driver.session(database=database) as session:
        state = session.run('MATCH (s:GraphIntelRegistry {id: $id}) RETURN s.version AS version',
                            id='phase1').single()
        entities = [dict(row['props']) for row in session.run('MATCH (e:Entity) RETURN properties(e) AS props')]
    if entities and state is None:
        raise ValueError('existing graph has no registry version: explicit migration required')
    return {'version': state['version'] if state else 0, 'target': target, 'entities': entities}


def write_graph(driver, database, target, records, global_resolution, evidence_reviews,
                authorized=False, schema_authorized=False):
    """No retries: a lost commit acknowledgement needs DB reconciliation.

    DDL constraints use separate transactions and may remain if data rolls back.
    The entire data change is one explicit transaction. Never execute user/model
    supplied Cypher: rebuild the plan from reviewed extraction records here.
    """
    if not authorized or not database or not target:
        raise ValueError('explicit graph write authority and target required')
    if global_resolution.get('registry_target') != target:
        raise ValueError('resolution was made against another target')
    plan = graph_plan(records, [], evidence_reviews, global_resolution)
    if plan['registry_version'] is None:
        raise ValueError('global registry snapshot required before graph writes')
    schema = plan['statements'][:3]
    data = plan['statements'][3:]
    with driver.session(database=database) as session:
        if schema_authorized:
            for statement in schema:
                session.run(statement['query'], statement['parameters']).consume()
            session.run('CREATE CONSTRAINT registry_id IF NOT EXISTS FOR (s:GraphIntelRegistry) REQUIRE s.id IS UNIQUE').consume()
        # require installed constraints rather than silently doing unapproved DDL
        required = {'entity_id': ('Entity', 'id'), 'mention_id': ('Mention', 'id'),
                    'citation_id': ('Citation', 'id'), 'registry_id': ('GraphIntelRegistry', 'id')}
        constraints = {}
        for row in session.run('SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties RETURN name, type, labelsOrTypes, properties'):
            if row['type'] in ('UNIQUENESS', 'NODE_PROPERTY_UNIQUENESS'):
                constraints[row['name']] = (tuple(row['labelsOrTypes']), tuple(row['properties']))
        if any(constraints.get(name) != ((label,), (prop,)) for name, (label, prop) in required.items()):
            raise ValueError('required uniqueness constraints missing; schema setup authority required')
        with session.begin_transaction() as tx:
            row = tx.run('MERGE (s:GraphIntelRegistry {id: $id}) ON CREATE SET s.version = 0 '
                         'SET s.locked = true RETURN s.version AS version', id='phase1').single()
            if row['version'] != plan['registry_version']:
                raise ValueError('graph registry changed after resolution; re-read and re-resolve')
            for statement in data:
                tx.run(statement['query'], statement['parameters']).consume()
            tx.run('MATCH (s:GraphIntelRegistry {id: $id}) SET s.version = s.version + 1, s.locked = false',
                   id='phase1').consume()
            tx.commit()  # Unknown acknowledgement is NOT permission to retry.
    after = read_registry(driver, database, target)
    expected = {e['id']: e for e in global_resolution['canonical_entities']}
    observed = {e['id']: e for e in after['entities']}
    verified = (after['version'] == plan['registry_version'] + 1
                and all(observed.get(key) == value for key, value in expected.items()))
    edge_checks = []
    for statement in data:
        params = statement['parameters']
        if 'citation_id' in params:
            edge_checks.append({'id': params['citation_id'], 'source_id': params['source_id'],
                                'target_id': params['target_id'],
                                'relationship_type': params['props']['relationship_type']})
    with driver.session(database=database) as session:
        found = {row['id'] for row in session.run(
            'UNWIND $edges AS edge MATCH (s:Entity {id: edge.source_id})-[r]->(t:Entity {id: edge.target_id}) '
            'WHERE r.id = edge.id AND type(r) = edge.relationship_type RETURN r.id AS id', edges=edge_checks)}
    verified = verified and found == {e['id'] for e in edge_checks}
    return {'status': 'Committed and read-back verified' if verified else 'Committed; read-back mismatch needs reconciliation', 'registry_version': plan['registry_version'] + 1,
            'target': target, 'database': database, 'data_statements': len(data),
            'read_back_verified': verified, 'warning': 'A post-commit failure or mismatch never triggers an automatic write retry.'}
