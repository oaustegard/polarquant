"""polar-embed: Retrieval-validated embedding compression.

Compress embeddings 4-8x with proven recall. Based on the rotation + Lloyd-Max
scalar quantization insight from TurboQuant (Zandieh et al., ICLR 2026,
arXiv:2504.19874). Implements the MSE-optimal stage which empirically
outperforms the full TurboQuant Prod variant for nearest-neighbor retrieval.
"""

from polar_embed.core import PolarQuantizer, CompressedVectors
from polar_embed.codebook import lloyd_max_codebook

__version__ = "0.2.0"
__all__ = ["PolarQuantizer", "CompressedVectors", "lloyd_max_codebook"]
