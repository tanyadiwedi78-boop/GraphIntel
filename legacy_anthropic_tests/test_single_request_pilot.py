"""One explicitly approved actual-charge attempt. Fake transport only."""
import json
import pytest
from app.extraction.anthropic_batch import prepare_plan, submit_job, write_new, digest
from tests.test_anthropic_batch import ontology, FakeAPI

@pytest.fixture
def pilot(tmp_path, ontology):
    source = tmp_path / 'chunks.jsonl'
    source.write_text(json.dumps({'chunk_id':'approved-source', 'text':'Cover page.',
                                 'accession':'a1', 'form':'8-K', 'period':'2026'}))
    cache = tmp_path / 'cache'
    plan = prepare_plan(source, ontology, 1, cache_dir=cache)
    folder = tmp_path / 'job'
    folder.mkdir()
    write_new(folder / 'plan.json', plan)
    return folder, plan

def send(pilot, api, **changes):
    folder, plan = pilot
    args = dict(job_dir=folder, approved_plan=plan['plan_id'], budget_usd=None,
                allow_paid=True, api=api, max_calls=1, single_request_actual_charge=True,
                approved_chunk_id='approved-source')
    args.update(changes)
    return submit_job(**args)

def test_one_post_claim_and_unchanged_plan(pilot):
    folder, plan = pilot
    before = (folder/'plan.json').read_bytes()
    api = FakeAPI([{'id':'msgbatch_once'}])
    assert send(pilot,api)['id'] == 'msgbatch_once'
    assert len(api.calls)==1 and api.calls[0][0]=='POST'
    assert api.calls[0][2] == {'requests':plan['requests']}
    claim = json.loads((folder/'submission-claim.json').read_text())
    assert claim['budget_usd'] is None and claim['hard_usd_cap'] is None
    assert claim['admission_mode']=='single_request_actual_charge_v1'
    assert claim['approved_chunk_id']=='approved-source'
    assert claim['custom_id']==plan['requests'][0]['custom_id']
    assert (folder/'plan.json').read_bytes()==before
    with pytest.raises(ValueError,match='already has'):
        send(pilot,api)
    assert len(api.calls)==1

@pytest.mark.parametrize('change', [dict(allow_paid=False), dict(approved_plan='wrong'),
    dict(max_calls=0), dict(approved_chunk_id=None), dict(approved_chunk_id='another'),
    dict(budget_usd='0.10'), dict(single_request_actual_charge=False)])
def test_wrong_or_missing_scope_stops_before_network(pilot,change):
    api=FakeAPI([])
    with pytest.raises(ValueError): send(pilot,api,**change)
    assert not api.calls
    assert not (pilot[0]/'submission-claim.json').exists()

@pytest.mark.parametrize('field,value', [('round',1),('parent_plan_id','parent'),
    ('root_plan_id','root'),('feedback',{'x':'repair'})])
def test_repairs_not_admitted(pilot,field,value):
    folder,plan=pilot
    plan[field]=value
    plan.pop('plan_id')
    plan['plan_id']=digest(plan)
    (folder/'plan.json').unlink()
    write_new(folder/'plan.json',plan)
    api=FakeAPI([])
    with pytest.raises(ValueError):send(pilot,api)
    assert not api.calls

@pytest.mark.parametrize('artifact',['submission-claim.json','submission-response.txt',
    'receipt.json','provider-results.jsonl','summary.json'])
def test_existing_job_state_blocks(pilot,artifact):
    (pilot[0]/artifact).write_text('{}')
    api=FakeAPI([])
    with pytest.raises(ValueError,match='already has'):send(pilot,api)
    assert not api.calls

def test_global_claim_blocks_even_same_plan(pilot):
    folder,plan=pilot
    claims=__import__('pathlib').Path(plan['cache_dir'])/'anthropic_claims'
    claims.mkdir(parents=True)
    write_new(claims/(plan['requests'][0]['custom_id']+'.json'),{'plan_id':plan['plan_id']})
    api=FakeAPI([])
    with pytest.raises(ValueError,match='already has'):send(pilot,api)
    assert not api.calls

def test_timeout_keeps_claim_and_blocks_retry(pilot):
    api=FakeAPI([TimeoutError()])
    with pytest.raises(TimeoutError):send(pilot,api)
    assert (pilot[0]/'submission-claim.json').exists()
    with pytest.raises(ValueError):send(pilot,api)
    assert len(api.calls)==1

def test_cli_rejects_both_admission_modes():
    from app.extraction.anthropic_batch import main
    with pytest.raises(SystemExit):
        main(['submit','--job-dir','unused','--approve-plan','unused',
              '--budget-usd','1','--single-request-actual-charge','--allow-paid'])

def test_multiple_requests_rejected(tmp_path,ontology):
    source=tmp_path/'chunks.jsonl'
    source.write_text('\n'.join(json.dumps({'chunk_id':'c'+str(i),'text':'Cover.',
        'accession':'a','form':'8-K','period':'2026'}) for i in range(2)))
    plan=prepare_plan(source,ontology,2,cache_dir=tmp_path/'cache')
    folder=tmp_path/'job';folder.mkdir();write_new(folder/'plan.json',plan)
    api=FakeAPI([])
    with pytest.raises(ValueError,match='one original request'):
        send((folder,plan),api,max_calls=2)
    assert not api.calls

def test_mutated_plan_stops(pilot):
    folder,plan=pilot
    plan['requests'][0]['params']['max_tokens']=1
    (folder/'plan.json').write_text(json.dumps(plan))
    api=FakeAPI([])
    with pytest.raises(ValueError,match='plan changed'):send(pilot,api)
    assert not api.calls

def test_cached_result_stops(pilot,monkeypatch):
    monkeypatch.setattr('app.extraction.anthropic_batch.compatible_cache',lambda *args: True)
    api=FakeAPI([])
    with pytest.raises(ValueError,match='already valid in cache'):send(pilot,api)
    assert not api.calls

def test_global_claim_records_pilot_scope(pilot):
    from pathlib import Path
    folder,plan=pilot
    send(pilot,FakeAPI([{'id':'msgbatch_once'}]))
    path=Path(plan['cache_dir'])/'anthropic_claims'/(plan['requests'][0]['custom_id']+'.json')
    claim=json.loads(path.read_text())
    assert claim['admission_mode']=='single_request_actual_charge_v1'
    assert claim['approved_chunk_id']=='approved-source'
    assert claim['hard_usd_cap'] is None
