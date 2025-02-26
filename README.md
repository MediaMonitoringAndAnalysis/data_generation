# Data Generation

A Python library for data generation using RAG (Retrieval-Augmented Generation) techniques.

## Installation

```bash
pip install git+https://github.com/MediaMonitoringAndAnalysis/data_generation.git
```

## Usage
```python
from data_generation import RAG
import pandas as pd
import torch

df = pd.DataFrame(...) # Your data with embeddings
question_embeddings = {...} # Your question embeddings
results = RAG(
    df=df,
    question_embeddings=question_embeddings,
    n_kept_entries=10,
    country="Country Name",
    language="english"
)
```

## License

This project is licensed under the GNU Affero General Public License v3 - see the LICENSE file for details.