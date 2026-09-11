from typing import Any, Dict, List

# Evaluation and self-test dataset with sample PDFs and expected sentences
SELFTEST_DATASET: List[Dict[str, Any]] = [
    {
        "filename": "studyboard.pdf",
        "sentences": [
            "Finn and Tyge were present.",
            "Poul was missing.",
            "We discussed the issue of calculators.",
        ],
    },
    {
        "filename": "2303.15133.pdf",
        "sentences": [
            "Other endpoints than the configured default can be queried.",
            "I call the tool Synia with the canonical homepage set up at https://synia.toolforge.org/.",
            "Scholia is a Web application running from the Wikimedia Foundation Toolforge server at http://scholia.toolforge.org.",
        ],
    },
]

