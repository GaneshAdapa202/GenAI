import sys
import warnings
from datetime import datetime   # ✅ must be here

from demo.crew import Demo

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI LLMs',
        'current_year': str(datetime.now().year)
    }

    try:
        Demo().crew(
            topic=inputs['topic'], 
            current_year=inputs['current_year']
        ).kickoff()
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
