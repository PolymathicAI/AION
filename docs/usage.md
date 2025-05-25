# Usage Guide

Load the pretrained model and start experimenting with astronomical data:

```python
from aion import AION

model = AION.from_pretrained('aion-base')
```

The model accepts modality-specific tokenized inputs. Refer to the API documentation for details on available modalities and helper functions.
