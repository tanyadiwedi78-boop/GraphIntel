"""Start the Gemini/OpenAI Phase1 pipeline. No historical provider imports."""
import sys
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_folder))

if __name__ == '__main__':
    arguments = sys.argv[1:]

    if not arguments or arguments == ['--help'] or arguments == ['-h']:
        print('No extraction started.')
        print('To check the pipeline, run:')
        print('python scripts/run_extraction.py phase1 --help')
        raise SystemExit(0)

    if arguments[0] != 'phase1':
        print('Use phase1. Historical provider commands are not part of this entry point.')
        raise SystemExit(2)

    from app.extraction.pipeline import main
    raise SystemExit(main(arguments[1:]))
