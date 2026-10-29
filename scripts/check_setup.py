import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, load_settings  
from app.corpus.data_limits import PlanError, load_plan  
from app.graph.ontology import OntologyError, load_ontology  



def main() -> int:
    try:
        settings = load_settings()
        ontology = load_ontology(settings.ontology_path)
        plan = load_plan(settings.companies_path)
    except (ConfigError, OntologyError, PlanError) as exc:
        print(f"PROBLEM: {exc}")
        return 1

    print(f"Settings . Model calls allowed per run: {settings.max_extraction_calls}")
    print(f"Ontology : {len(ontology.entity_types)} entity types, {len(ontology.relations)} relations")
    print(f"fingerprint {ontology.fingerprint()}")
    print(f"Edge edge carries: {', '.join(ontology.edge_fields)}")
    print(
        f"Plan OK: {len(plan.companies)} companies, forms {', '.join(plan.forms)}, "
        f"{plan.first_year} to {plan.last_year} ({plan.years} years)"
    )
    missing = plan.missing_ciks()
    if missing:
        print(f"  CIK empty for: {', '.join(missing)} ")
    else:
        print(" Companies_Sec Fillings Number (CIK)")
    if "@" not in settings.edgar_user_agent:
        print("Error : EDGAR_USER_AGENT missing (.env)")
        return 1
    print("EDGAR_USER_AGENT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
